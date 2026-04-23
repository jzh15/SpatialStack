import logging
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("lmms-eval")


def _ensure_spatialrgpt_on_path() -> None:
    repo_path = os.environ.get("SPATIALRGPT_PATH", "/workspace/SpatialRGPT")
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)


_ensure_spatialrgpt_on_path()

try:
    from llava.constants import (
        DEFAULT_DEPTH_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_MASK_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        process_regions,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except Exception as e:
    eval_logger.debug(f"SpatialRGPT is not available on PYTHONPATH. Error: {e}")


def prepare_config_for_eval(config, kwargs):
    # Compatible with deprecated config convention
    if getattr(config, "vision_tower_cfg", None) is None:
        config.vision_tower_cfg = config.mm_vision_tower

    config.model_dtype = kwargs.pop("torch_dtype").__str__()


# Patch builder to use evaluation config helper (mirrors lmms_eval.models.vila)
try:
    import llava

    llava.model.builder.prepare_config_for_eval = prepare_config_for_eval
except Exception:
    pass


@register_model("spatialrgpt")
class SpatialRGPT(lmms):
    _depth_model = None
    _depth_transform = None
    _depth_device = None

    def __init__(
        self,
        pretrained: str = "/workspace/models/SpatialRGPT-VILA1.5-8B",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[int] = 1,
        conv_template: str = "llama_3",
        device_map: Optional[str] = None,
        use_depth: Optional[bool] = False,
        depth_anything_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self._device = torch.device(device)
        self.device_map = device_map or device

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained,
            self.model_name,
            device_map=self.device_map,
        )
        self._config = self._model.config
        self.conv_template = conv_template
        self.batch_size_per_gpu = int(batch_size)
        self.use_depth = str(use_depth).lower() in ("1", "true", "yes")
        self.depth_anything_path = depth_anything_path or os.environ.get("DEPTH_ANYTHING_PATH")

        self.model.eval()
        if device_map != "auto":
            self.model.to(self._device)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("SpatialRGPT loglikelihood is not implemented for lmms_eval.")

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if visuals in (None, [None]):
                images = None
            else:
                if not isinstance(visuals, list):
                    visuals = [visuals]
                images = visuals

            if images is None:
                qs = contexts
                images_tensor = None
                depths_tensor = None
                masks_tensor = None
            else:
                if not all(isinstance(img, Image.Image) for img in images):
                    raise NotImplementedError("SpatialRGPT only supports image inputs for now.")
                images_tensor = process_images(images, self._image_processor, self.model.config).to(
                    self.device, dtype=torch.float16
                )
                if self.model.config.mm_use_im_start_end:
                    image_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                else:
                    image_tokens = (DEFAULT_IMAGE_TOKEN + "\n") * len(images)
                qs = f"{image_tokens}\n{contexts}"

                depths_tensor = None
                masks_tensor = None
                if self.use_depth:
                    depth_model, depth_transform = self._get_depth_predictor()
                    # full-image mask + depth map per image
                    masks = []
                    depth_images = []
                    for img in images:
                        raw = np.array(img.convert("RGB"))
                        h, w = raw.shape[:2]
                        masks.append(np.ones((h, w), dtype=np.uint8))

                        depth_input = depth_transform({"image": raw / 255.0})["image"]
                        depth_input = torch.from_numpy(depth_input).unsqueeze(0).to(self.device)
                        raw_depth = depth_model(depth_input)
                        raw_depth = F.interpolate(raw_depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
                        raw_depth = raw_depth.detach().cpu().numpy()
                        raw_depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
                        raw_depth = raw_depth.astype(np.uint8)
                        colorized_depth = np.stack([raw_depth, raw_depth, raw_depth], axis=-1)
                        depth_images.append(Image.fromarray(colorized_depth))

                    masks_tensor = process_regions(masks, self._image_processor, self.model.config).to(
                        self.device, dtype=torch.float16
                    )
                    depths_tensor = process_images(depth_images, self._image_processor, self.model.config).to(
                        self.device, dtype=torch.float16
                    )
                    qs = f"{qs}\n{DEFAULT_MASK_TOKEN} {DEFAULT_DEPTH_TOKEN}"

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
            input_ids = input_ids.to(self.device)
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            attention_mask = input_ids.ne(pad_token_id).long()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 64
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=[images_tensor] if images_tensor is not None else None,
                    depths=[depths_tensor] if depths_tensor is not None else None,
                    masks=[masks_tensor] if masks_tensor is not None else None,
                    do_sample=gen_kwargs.get("do_sample", False),
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)].strip()

            res.append(outputs)
            self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), outputs)
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("SpatialRGPT multi-round generation is not implemented.")

    def _get_depth_predictor(self):
        if not self.depth_anything_path:
            raise RuntimeError("DEPTH_ANYTHING_PATH is not set; cannot run depth-aug evaluation.")

        if (
            SpatialRGPT._depth_model is not None
            and SpatialRGPT._depth_transform is not None
            and SpatialRGPT._depth_device == self.device
        ):
            return SpatialRGPT._depth_model, SpatialRGPT._depth_transform

        sys.path.append(self.depth_anything_path)
        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
        from torchvision.transforms import Compose

        ckpt_path = os.path.join(self.depth_anything_path, "checkpoints", "depth_anything_vitl14.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Depth-Anything checkpoint not found: {ckpt_path}")

        depth_model = DepthAnything(
            {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024], "localhub": False}
        )
        depth_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        depth_model = depth_model.to(self.device).eval()

        depth_transform = Compose(
            [
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        SpatialRGPT._depth_model = depth_model
        SpatialRGPT._depth_transform = depth_transform
        SpatialRGPT._depth_device = self.device
        return depth_model, depth_transform
