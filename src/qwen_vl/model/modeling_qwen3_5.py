import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5DynamicCache,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
)

from .feature_fusion import (
    FeatureFusionConfig,
    FeatureFusionModule,
    GeometryFeatureMerger,
    MultiLayerFeatureFusionConfig,
    MultiLayerFeatureFusionModule,
)
from .geometry_encoders import GeometryEncoderConfig, create_geometry_encoder
from .position_utils import get_2d_sincos_pos_embed


GEOMETRY_STATE_KEYWORDS = (
    "geometry_encoder",
    "language_feature_fusion",
    "feature_fusion",
    "geometry_merger",
)


def move_qwen3_5_geometry_modules_to_device(
    geometry_encoder: Optional[nn.Module],
    language_feature_fusion: Optional[nn.Module],
    feature_fusion: Optional[nn.Module],
    geometry_merger: Optional[nn.Module],
    geometry_merger_list: Optional[nn.Module],
    device: Optional[torch.device],
    dtype: Optional[torch.dtype] = None,
):
    if device is None or device.type == "meta":
        return
    if geometry_encoder is not None and hasattr(geometry_encoder, "to"):
        geometry_encoder.to(device=device)
    if language_feature_fusion is not None and hasattr(language_feature_fusion, "to"):
        if dtype is not None:
            language_feature_fusion.to(device=device, dtype=dtype)
        else:
            language_feature_fusion.to(device=device)
    if feature_fusion is not None and hasattr(feature_fusion, "to"):
        if dtype is not None:
            feature_fusion.to(device=device, dtype=dtype)
        else:
            feature_fusion.to(device=device)
    if geometry_merger is not None and hasattr(geometry_merger, "to"):
        if dtype is not None:
            geometry_merger.to(device=device, dtype=dtype)
        else:
            geometry_merger.to(device=device)
    if geometry_merger_list is not None and hasattr(geometry_merger_list, "to"):
        if dtype is not None:
            geometry_merger_list.to(device=device, dtype=dtype)
        else:
            geometry_merger_list.to(device=device)


def align_qwen3_5_geometry_modules(model):
    inner_model = getattr(model, "model", None)
    if inner_model is None:
        return model

    reference_tensor = getattr(getattr(model, "lm_head", None), "weight", None)
    if reference_tensor is None or reference_tensor.device.type == "meta":
        for module_name in ("language_model", "visual"):
            module = getattr(inner_model, module_name, None)
            if module is None:
                continue
            try:
                reference_tensor = next(module.parameters())
            except StopIteration:
                continue
            if reference_tensor.device.type != "meta":
                break

    device = getattr(reference_tensor, "device", None)
    dtype = getattr(reference_tensor, "dtype", None)
    move_qwen3_5_geometry_modules_to_device(
        getattr(inner_model, "geometry_encoder", None),
        getattr(inner_model, "language_feature_fusion", None),
        getattr(inner_model, "feature_fusion", None),
        getattr(inner_model, "geometry_merger", None),
        getattr(inner_model, "geometry_merger_list", None),
        device,
        dtype,
    )
    model.geometry_encoder = getattr(inner_model, "geometry_encoder", None)
    model.language_feature_fusion = getattr(inner_model, "language_feature_fusion", None)
    model.feature_fusion = getattr(inner_model, "feature_fusion", None)
    model.geometry_merger = getattr(inner_model, "geometry_merger", None)
    model.geometry_merger_list = getattr(inner_model, "geometry_merger_list", None)
    return model


def _iter_qwen3_5_checkpoint_files(pretrained_model_name_or_path: str) -> List[Path]:
    checkpoint_path = Path(pretrained_model_name_or_path)
    if checkpoint_path.is_file():
        return [checkpoint_path]
    if not checkpoint_path.is_dir():
        return []

    direct_candidates = [
        checkpoint_path / "model.safetensors",
        checkpoint_path / "pytorch_model.bin",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return [candidate]

    index_candidates = [
        checkpoint_path / "model.safetensors.index.json",
        checkpoint_path / "pytorch_model.bin.index.json",
    ]
    for index_file in index_candidates:
        if not index_file.exists():
            continue
        with index_file.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
        files = []
        for key, filename in weight_map.items():
            if any(keyword in key for keyword in GEOMETRY_STATE_KEYWORDS):
                files.append(checkpoint_path / filename)
        return sorted(set(files))

    return []


def _resolve_qwen3_5_checkpoint_root(pretrained_model_name_or_path: str) -> str:
    checkpoint_path = Path(pretrained_model_name_or_path)
    if checkpoint_path.exists():
        return str(checkpoint_path)

    try:
        cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE")
        return snapshot_download(
            pretrained_model_name_or_path,
            repo_type="model",
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception:
        return pretrained_model_name_or_path


def _infer_qwen3_5_language_fusion_geo_hidden_size(
    sub_state_dict: Dict[str, torch.Tensor],
    spatial_merge_size: int,
) -> Optional[int]:
    candidate_sizes = []
    merged_patch_size = spatial_merge_size**2

    for key, tensor in sub_state_dict.items():
        if "language_feature_fusion" not in key:
            continue
        if key.endswith("geo_ln.weight") or key.endswith("geo_ln.bias"):
            candidate_sizes.append(int(tensor.shape[0]))
            continue
        if key.endswith("geo_mlp.0.weight") and tensor.ndim == 2:
            input_dim = int(tensor.shape[1])
            if input_dim % merged_patch_size == 0:
                candidate_sizes.append(input_dim // merged_patch_size)

    if not candidate_sizes:
        return None

    inferred_size = candidate_sizes[0]
    if any(size != inferred_size for size in candidate_sizes[1:]):
        return None
    return inferred_size


def _validate_qwen3_5_vision_language_fusion_checkpoint(
    model,
    sub_state_dict: Dict[str, torch.Tensor],
) -> None:
    config = getattr(model, "config", None)
    if getattr(config, "feature_fusion_method", None) != "deepstack_language_add":
        return
    if not getattr(config, "vision_language_fusion_layers", None):
        return

    inner_model = getattr(model, "model", model)
    current_fusion = getattr(inner_model, "language_feature_fusion", None)
    if current_fusion is None:
        return

    current_config = getattr(current_fusion, "config", None)
    spatial_merge_size = getattr(current_config, "spatial_merge_size", None)
    if spatial_merge_size is None:
        return

    inferred_geo_hidden_size = _infer_qwen3_5_language_fusion_geo_hidden_size(sub_state_dict, spatial_merge_size)
    expected_geo_hidden_size = getattr(current_fusion, "geo_hidden_size", None)
    if inferred_geo_hidden_size is None or expected_geo_hidden_size is None:
        return
    if inferred_geo_hidden_size == expected_geo_hidden_size:
        return

    raise ValueError(
        "Checkpoint language_feature_fusion weights are incompatible with Qwen3.5 vision-language fusion "
        f"(checkpoint geo_hidden_size={inferred_geo_hidden_size}, expected={expected_geo_hidden_size}). "
        "This checkpoint was likely trained without vision_language_fusion_layers applied and actually learned "
        "geometry-fusion weights. Retrain it for vision fusion, or evaluate it as a geometry-fusion checkpoint."
    )


def _load_qwen3_5_geometry_submodules(model, pretrained_model_name_or_path: str) -> int:
    checkpoint_files = _iter_qwen3_5_checkpoint_files(pretrained_model_name_or_path)
    if not checkpoint_files:
        return 0

    model_keys = set(model.state_dict().keys())
    loaded_key_count = 0

    for checkpoint_file in checkpoint_files:
        suffixes = checkpoint_file.suffixes
        if suffixes and suffixes[-1] == ".safetensors":
            with safe_open(checkpoint_file, framework="pt", device="cpu") as handle:
                sub_state_dict = {
                    key: handle.get_tensor(key)
                    for key in handle.keys()
                    if key in model_keys and any(keyword in key for keyword in GEOMETRY_STATE_KEYWORDS)
                }
        else:
            state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
            sub_state_dict = {
                key: value
                for key, value in state_dict.items()
                if key in model_keys and any(keyword in key for keyword in GEOMETRY_STATE_KEYWORDS)
            }

        if not sub_state_dict:
            continue
        _validate_qwen3_5_vision_language_fusion_checkpoint(model, sub_state_dict)
        model.load_state_dict(sub_state_dict, strict=False)
        loaded_key_count += len(sub_state_dict)

    return loaded_key_count


class Qwen3_5TextModelWithGeometry(Qwen3_5TextModel):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        geometry_layer_features: Optional[Dict[int, List[torch.Tensor]]] = None,
        fusion_module: Optional[nn.Module] = None,
        image_mask: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        include_camera_token: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        vis_pos_embed_per_image: Optional[torch.Tensor] = None
        geo_pos_embed_per_image: Optional[torch.Tensor] = None
        if geometry_layer_features and fusion_module is not None and grid_thw is not None:
            num_images = len(grid_thw)
            if num_images > 0:
                num_grid_h, num_grid_w = grid_thw[0][1:].tolist()
                merge_size = fusion_module.config.spatial_merge_size
                num_merged_grid_h = num_grid_h // merge_size
                num_merged_grid_w = num_grid_w // merge_size
                lang_hidden_size = getattr(fusion_module, "lang_hidden_size")

                pos_grid = get_2d_sincos_pos_embed(
                    num_merged_grid_h + 1,
                    num_merged_grid_w + 1,
                    lang_hidden_size,
                    hidden_states.device,
                ).to(hidden_states.dtype)
                pos_grid = pos_grid.reshape(num_merged_grid_h + 1, num_merged_grid_w + 1, -1)

                row_idx = torch.arange(num_merged_grid_h, device=hidden_states.device)
                col_idx = torch.arange(num_merged_grid_w, device=hidden_states.device)
                grid_y, grid_x = torch.meshgrid(row_idx, col_idx, indexing="ij")
                patch_pos = torch.stack([grid_y, grid_x], dim=-1)
                patch_pos = patch_pos.unsqueeze(0).repeat(num_images, 1, 1, 1)
                patch_pos = patch_pos.reshape(num_images, num_merged_grid_h * num_merged_grid_w, -1)
                patch_pos = patch_pos + 1

                h_idx_vis, w_idx_vis = patch_pos.unbind(dim=-1)
                h_idx_vis = h_idx_vis.to(dtype=torch.long, device=pos_grid.device)
                w_idx_vis = w_idx_vis.to(dtype=torch.long, device=pos_grid.device)
                vis_pos_embed_per_image = pos_grid[h_idx_vis, w_idx_vis]

                if include_camera_token:
                    cam_embed = pos_grid[0, 0].unsqueeze(0).unsqueeze(0).repeat(num_images, 1, 1)
                    geo_pos_embed_per_image = torch.cat([cam_embed, vis_pos_embed_per_image], dim=1)
                else:
                    geo_pos_embed_per_image = vis_pos_embed_per_image

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            if (
                geometry_layer_features is not None
                and fusion_module is not None
                and image_mask is not None
                and layer_idx in geometry_layer_features
            ):
                vision_token_mask = image_mask[..., 0]
                vision_tokens = hidden_states[vision_token_mask]
                fused = fusion_module(
                    vision_tokens,
                    geometry_layer_features[layer_idx],
                    layer_idx,
                    vis_pos_embed_per_image,
                    geo_pos_embed_per_image,
                )
                hidden_states = hidden_states.clone()
                hidden_states[vision_token_mask] = fused

        hidden_states = self.norm(hidden_states)

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3_5ModelWithGeometry(Qwen3_5Model):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = Qwen3_5TextModelWithGeometry(config.text_config)
        self.geometry_encoder = None
        self.language_feature_fusion = None
        self.feature_fusion = None
        self.geometry_merger = None
        self.geometry_merger_list = None
        self._geometry_modules_initialized = False

        if getattr(config, "use_geometry_encoder", False):
            self._validate_geometry_config(config)
            if not self._should_defer_geometry_init():
                self.initialize_geometry_modules()

    def _should_defer_geometry_init(self) -> bool:
        try:
            if torch.empty(0).device.type == "meta":
                return True
        except Exception:
            pass

        try:
            return next(self.visual.parameters()).device.type == "meta"
        except StopIteration:
            return False

    def _validate_geometry_config(self, config):
        fusion_method = getattr(config, "feature_fusion_method", "deepstack_language_add")
        if fusion_method == "deepstack_language_add":
            if not getattr(config, "geometry_fusion_layers", None):
                raise ValueError("Qwen3.5 geometry fusion requires geometry_fusion_layers to be set.")
            if (
                not getattr(config, "geometry_encoder_layers", None)
                and not getattr(config, "vision_language_fusion_layers", None)
            ):
                raise ValueError("Qwen3.5 geometry fusion requires geometry_encoder_layers to be set.")
            return

        if "deepstack" in fusion_method:
            raise NotImplementedError(
                "Qwen3.5 geometry currently supports deepstack_language_add or post-merger fusion methods, "
                f"got: {fusion_method}"
            )
        if not getattr(config, "geometry_encoder_layers", None):
            raise ValueError("Qwen3.5 post-merger geometry fusion requires geometry_encoder_layers to be set.")

    def initialize_geometry_modules(self):
        if self._geometry_modules_initialized:
            return

        config = self.config
        self._validate_geometry_config(config)
        fusion_method = getattr(config, "feature_fusion_method", "deepstack_language_add")

        encoder_config = GeometryEncoderConfig(
            encoder_type=getattr(config, "geometry_encoder_type", "vggt"),
            model_path=getattr(config, "geometry_encoder_path", None),
            reference_frame=getattr(config, "reference_frame", "first"),
            freeze_encoder=getattr(config, "geometry_encoder_freeze", True),
        )

        self.geometry_encoder = create_geometry_encoder(
            encoder_type=encoder_config.encoder_type,
            model_path=encoder_config.model_path,
            reference_frame=encoder_config.reference_frame,
            freeze_encoder=encoder_config.freeze_encoder,
        )

        if fusion_method == "deepstack_language_add":
            use_vision_language_fusion = getattr(config, "vision_language_fusion_layers", None) is not None
            fusion_config = MultiLayerFeatureFusionConfig(
                fusion_method=fusion_method,
                vis_hidden_size=self.visual.config.hidden_size,
                geo_hidden_size=(
                    self.visual.config.hidden_size
                    if use_vision_language_fusion
                    else self.geometry_encoder.get_feature_dim()
                ),
                lang_hidden_size=config.text_config.hidden_size,
                geometry_fusion_layers=getattr(config, "geometry_fusion_layers", None),
                pos_encoding_type=getattr(config, "pos_encoding_type", "none"),
                spatial_merge_size=config.vision_config.spatial_merge_size,
                num_heads=getattr(config, "fusion_attention_heads", 8),
                dropout=getattr(config, "fusion_dropout", 0.1),
            )
            self.language_feature_fusion = MultiLayerFeatureFusionModule(fusion_config)
            self.language_feature_fusion.apply(self._init_weights)
            self.language_feature_fusion.reset_residual_branches_to_noop()
        else:
            geometry_encoder_layers = getattr(config, "geometry_encoder_layers", None)
            merger_kwargs = dict(
                output_dim=config.text_config.hidden_size,
                hidden_dim=getattr(config, "geometry_merger_hidden_dim", 4096),
                context_dim=self.geometry_encoder.get_feature_dim(),
                spatial_merge_size=config.vision_config.spatial_merge_size,
                merger_type=getattr(config, "geometry_merger_type", "mlp"),
            )
            if geometry_encoder_layers is not None and len(geometry_encoder_layers) > 1:
                self.geometry_merger_list = nn.ModuleList(
                    GeometryFeatureMerger(**merger_kwargs) for _ in geometry_encoder_layers
                )
            else:
                self.geometry_merger = GeometryFeatureMerger(**merger_kwargs)

            fusion_config = FeatureFusionConfig(
                fusion_method=fusion_method,
                hidden_size=config.text_config.hidden_size,
                num_heads=getattr(config, "fusion_attention_heads", 8),
                dropout=getattr(config, "fusion_dropout", 0.1),
                num_layers=getattr(config, "fusion_num_layers", 1),
            )
            self.feature_fusion = FeatureFusionModule(fusion_config)
        self._geometry_modules_initialized = True

    def align_geometry_modules(self, reference_tensor: Optional[torch.Tensor] = None):
        if reference_tensor is None:
            try:
                reference_tensor = next(self.language_model.parameters())
            except StopIteration:
                reference_tensor = None

        device = getattr(reference_tensor, "device", None)
        dtype = getattr(reference_tensor, "dtype", None)
        move_qwen3_5_geometry_modules_to_device(
            getattr(self, "geometry_encoder", None),
            getattr(self, "language_feature_fusion", None),
            getattr(self, "feature_fusion", None),
            getattr(self, "geometry_merger", None),
            getattr(self, "geometry_merger_list", None),
            device,
            dtype,
        )

    def _process_geometry_features(self, image_embeds, geometry_encoder_inputs, geometry_encoder_layers=[-2]):
        batch_size = len(geometry_encoder_inputs)
        geo_embeds = []
        for bn in range(batch_size):
            if geometry_encoder_inputs[bn].shape[0] > 0:
                n_image, _, height, width = geometry_encoder_inputs[bn].shape
                features = self.geometry_encoder.encode_layers(
                    geometry_encoder_inputs[bn],
                    layer_indices=geometry_encoder_layers,
                )[-1].to(image_embeds.dtype)
                features = features.reshape(
                    n_image,
                    height // self.geometry_encoder.patch_size,
                    width // self.geometry_encoder.patch_size,
                    -1,
                )
                features = self.geometry_merger(features)
                geo_embeds.append(features)

        geo_embeds = torch.cat(geo_embeds, dim=0) if geo_embeds else None
        if geo_embeds is not None:
            image_embeds = image_embeds.view(geo_embeds.shape)
            image_embeds = self.feature_fusion(image_embeds, geo_embeds)
            image_embeds = image_embeds.view(-1, image_embeds.shape[-1])
        return image_embeds

    def _process_multi_geometry_features(self, image_embeds, geometry_encoder_inputs, geometry_encoder_layers=[-2]):
        geo_feature_list = self.geometry_encoder.encode_layers(
            geometry_encoder_inputs[0],
            layer_indices=geometry_encoder_layers,
        )
        n_image, _, height, width = geometry_encoder_inputs[0].shape

        for geo_feature, geo_merger in zip(geo_feature_list, self.geometry_merger_list):
            geo_feature = geo_feature.reshape(
                n_image,
                height // self.geometry_encoder.patch_size,
                width // self.geometry_encoder.patch_size,
                -1,
            )
            geo_feature = geo_merger(geo_feature)

            image_embeds = image_embeds.view(geo_feature.shape)
            image_embeds = self.feature_fusion(image_embeds, geo_feature)
            image_embeds = image_embeds.view(-1, image_embeds.shape[-1])

        return image_embeds

    def _collect_geometry_layer_features(self, geometry_encoder_inputs):
        fusion_layers = getattr(self.config, "geometry_fusion_layers", None)
        geometry_encoder_layers = getattr(self.config, "geometry_encoder_layers", None)
        spatial_merge_size = getattr(self.config.vision_config, "spatial_merge_size", 2)
        include_camera_token = getattr(self.config, "include_camera_token", False)

        if geometry_encoder_inputs is None:
            return None
        if fusion_layers is None or geometry_encoder_layers is None:
            raise ValueError("Qwen3.5 geometry fusion requires both geometry_fusion_layers and geometry_encoder_layers.")
        assert len(geometry_encoder_inputs) == 1, "Qwen3.5 geometry fusion currently expects per-device batch size 1."
        layer_features = self.geometry_encoder.encode_layers(
            geometry_encoder_inputs[0],
            layer_indices=geometry_encoder_layers,
            spatial_merge_size=spatial_merge_size,
            include_camera_token=include_camera_token,
        )

        geometry_layer_features: Dict[int, List[torch.Tensor]] = {}
        for layer_idx, layer_feature in zip(fusion_layers, layer_features):
            geometry_layer_features.setdefault(layer_idx, []).append(layer_feature)

        return geometry_layer_features

    def _collect_vision_layer_features(self, vision_hidden_states, image_grid_thw):
        fusion_layers = getattr(self.config, "geometry_fusion_layers", None)
        vision_language_fusion_layers = getattr(self.config, "vision_language_fusion_layers", None)

        if fusion_layers is None or vision_language_fusion_layers is None:
            raise ValueError(
                "geometry_fusion_layers and vision_language_fusion_layers must be set for vision deepstack language fusion."
            )
        if len(fusion_layers) != len(vision_language_fusion_layers):
            raise ValueError(
                "vision_language_fusion_layers and geometry_fusion_layers must have the same length."
            )
        if vision_hidden_states is None or image_grid_thw is None:
            return None

        split_sizes = image_grid_thw.prod(-1).tolist()
        vision_layer_features: Dict[int, List[torch.Tensor]] = {}
        num_hidden_states = len(vision_hidden_states)
        for decoder_layer, vision_layer in zip(fusion_layers, vision_language_fusion_layers):
            if vision_layer >= num_hidden_states or vision_layer < -num_hidden_states:
                continue
            layer_feature = vision_hidden_states[vision_layer]
            if layer_feature is None:
                continue
            per_image_features = torch.stack(torch.split(layer_feature, split_sizes), dim=0)
            vision_layer_features.setdefault(decoder_layer, []).append(per_image_features)

        return vision_layer_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        geometry_encoder_inputs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Qwen3_5ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        fusion_method = getattr(self.config, "feature_fusion_method", "deepstack_language_add")
        use_language_deepstack = fusion_method == "deepstack_language_add"
        vision_language_fusion_layers = getattr(self.config, "vision_language_fusion_layers", None)
        should_capture_vision_layers = (
            vision_language_fusion_layers is not None
            and use_language_deepstack
            and (cache_position is None or (isinstance(cache_position, torch.Tensor) and cache_position[0] == 0))
        )

        image_mask = None
        vision_layer_features = None
        if pixel_values is not None:
            image_outputs = self.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True,
                output_hidden_states=should_capture_vision_layers,
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            if should_capture_vision_layers:
                vision_layer_features = getattr(image_outputs, "hidden_states", None)
            should_fuse_post_merger_geometry = (
                getattr(self.config, "use_geometry_encoder", False)
                and geometry_encoder_inputs is not None
                and not use_language_deepstack
                and (cache_position is None or (isinstance(cache_position, torch.Tensor) and cache_position[0] == 0))
            )
            if should_fuse_post_merger_geometry:
                if self.geometry_encoder is None or self.feature_fusion is None:
                    self.initialize_geometry_modules()
                self.align_geometry_modules(inputs_embeds)
                geometry_encoder_layers = getattr(self.config, "geometry_encoder_layers", None) or [-2]
                if len(geometry_encoder_layers) == 1:
                    image_embeds = self._process_geometry_features(
                        image_embeds,
                        geometry_encoder_inputs,
                        geometry_encoder_layers=geometry_encoder_layers,
                    )
                else:
                    image_embeds = self._process_multi_geometry_features(
                        image_embeds,
                        geometry_encoder_inputs,
                        geometry_encoder_layers=geometry_encoder_layers,
                    )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        geometry_layer_features = None
        fusion_module = None
        grid_thw = None
        include_camera_token = getattr(self.config, "include_camera_token", False)
        should_fuse_geometry = (
            getattr(self.config, "use_geometry_encoder", False)
            and use_language_deepstack
            and pixel_values is not None
            and (cache_position is None or (isinstance(cache_position, torch.Tensor) and cache_position[0] == 0))
            and (
                vision_layer_features is not None
                or (
                    getattr(self.config, "use_geometry_encoder", False)
                    and geometry_encoder_inputs is not None
                )
            )
        )
        if should_fuse_geometry:
            needs_geometry_encoder = vision_layer_features is None
            if (needs_geometry_encoder and self.geometry_encoder is None) or self.language_feature_fusion is None:
                self.initialize_geometry_modules()
            self.align_geometry_modules(inputs_embeds)
            if vision_layer_features is not None:
                include_camera_token = False
                geometry_layer_features = self._collect_vision_layer_features(
                    vision_layer_features,
                    image_grid_thw,
                )
            else:
                geometry_layer_features = self._collect_geometry_layer_features(geometry_encoder_inputs)
            fusion_module = self.language_feature_fusion
            grid_thw = image_grid_thw

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            geometry_layer_features=geometry_layer_features,
            fusion_module=fusion_module,
            image_mask=image_mask,
            grid_thw=grid_thw,
            include_camera_token=include_camera_token,
            **kwargs,
        )

        return Qwen3_5ModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


class Qwen3_5ForConditionalGenerationWithGeometry(Qwen3_5ForConditionalGeneration):
    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = Qwen3_5ModelWithGeometry(config)
        self.geometry_encoder = self.model.geometry_encoder
        self.language_feature_fusion = self.model.language_feature_fusion
        self.feature_fusion = self.model.feature_fusion
        self.geometry_merger = self.model.geometry_merger
        self.geometry_merger_list = self.model.geometry_merger_list
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
        if self.language_feature_fusion is not None:
            self.language_feature_fusion.reset_residual_branches_to_noop()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        geometry_encoder_path = kwargs.pop("geometry_encoder_path", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if getattr(model.config, "use_geometry_encoder", False):
            model.model.initialize_geometry_modules()
            align_qwen3_5_geometry_modules(model)
        if geometry_encoder_path and getattr(model.model, "geometry_encoder", None) is not None:
            model.model.geometry_encoder.load_model(geometry_encoder_path)
        if getattr(model.config, "use_geometry_encoder", False):
            resolved_checkpoint_root = _resolve_qwen3_5_checkpoint_root(pretrained_model_name_or_path)
            _load_qwen3_5_geometry_submodules(model, resolved_checkpoint_root)
        return align_qwen3_5_geometry_modules(model)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        geometry_encoder_inputs: Optional[List[torch.Tensor]] = None,
        tag: Optional[str] = None,
        **kwargs,
    ) -> Qwen3_5CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
            geometry_encoder_inputs=geometry_encoder_inputs,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
