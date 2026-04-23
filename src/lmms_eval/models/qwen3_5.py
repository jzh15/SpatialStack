import re
import time
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from packaging.version import Version
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_pil


MIN_QWEN3_5_TRANSFORMERS_VERSION = Version("5.3.0")


def require_qwen3_5_support():
    current_version = Version(transformers.__version__)
    if current_version < MIN_QWEN3_5_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Qwen3.5 evaluation requires transformers>="
            f"{MIN_QWEN3_5_TRANSFORMERS_VERSION}, but found {transformers.__version__}."
        )


def patch_qwen3_5_flash_attention():
    try:
        import transformers.modeling_flash_attention_utils as flash_attention_utils
    except ImportError:
        return

    if getattr(flash_attention_utils, "_spatialstack_qwen3_5_mrope_patch", False):
        return

    original_is_packed_sequence = flash_attention_utils._is_packed_sequence

    def patched_is_packed_sequence(position_ids, batch_size):
        if position_ids is not None and getattr(position_ids, "ndim", None) == 3:
            return False
        return original_is_packed_sequence(position_ids, batch_size)

    flash_attention_utils._is_packed_sequence = patched_is_packed_sequence
    flash_attention_utils._spatialstack_qwen3_5_mrope_patch = True


def is_image_path(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def strip_thinking_content(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def detect_qwen3_5_fast_path_runtime():
    runtime = {}
    for module_name in ("fla", "causal_conv1d"):
        try:
            __import__(module_name)
            runtime[module_name] = True
        except ImportError:
            runtime[module_name] = False
    return runtime


def build_qwen3_5_geometry_inputs(images, image_grid_thw, patch_size: int = 14):
    geometry_tensors = []
    max_height = 0
    max_width = 0

    for image, grid in zip(images, image_grid_thw):
        _, grid_h, grid_w = [int(v) for v in grid.tolist()]
        target_height = grid_h * patch_size
        target_width = grid_w * patch_size
        resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0
        geometry_tensors.append(tensor)
        max_height = max(max_height, target_height)
        max_width = max(max_width, target_width)

    padded_tensors = []
    for tensor in geometry_tensors:
        h_padding = max_height - tensor.shape[1]
        w_padding = max_width - tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = torch.nn.functional.pad(
                tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
        padded_tensors.append(tensor)

    return padded_tensors


def move_qwen3_5_eval_inputs_to_device(inputs, device):
    inputs = inputs.to(device)
    if "geometry_encoder_inputs" in inputs:
        inputs["geometry_encoder_inputs"] = [tensor.to(device) for tensor in inputs["geometry_encoder_inputs"]]
    return inputs


@register_model("qwen3_5")
class Qwen3_5(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        add_frame_index: bool = False,
        disable_thinking: bool = True,
        strip_thinking: bool = True,
        max_length: Optional[int] = None,
        geometry_encoder_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        require_qwen3_5_support()
        patch_qwen3_5_flash_attention()

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        self.max_num_frames = max_num_frames
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.add_frame_index = add_frame_index
        self.disable_thinking = disable_thinking
        self.strip_thinking = strip_thinking
        self.fast_path_runtime = detect_qwen3_5_fast_path_runtime()
        if not all(self.fast_path_runtime.values()):
            missing = ", ".join(name for name, available in self.fast_path_runtime.items() if not available)
            eval_logger.warning(
                f"Qwen3.5 optimized runtime dependencies are missing ({missing}). "
                "Upstream may fall back to slower torch kernels during eval."
            )

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = str(self._device)
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        config = AutoConfig.from_pretrained(pretrained)
        model_type = getattr(config, "model_type", None)
        if model_type not in {"qwen3_5", "qwen3_5_vl"}:
            raise ValueError(f"Unsupported model_type '{model_type}' for Qwen3.5 eval adapter.")
        use_geometry_model = getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False)
        if use_geometry_model and int(batch_size) != 1:
            raise ValueError("Qwen3.5 geometry evaluation currently requires batch_size=1.")

        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Your transformers build does not expose Qwen3_5ForConditionalGeneration. "
                f"Please install transformers>={MIN_QWEN3_5_TRANSFORMERS_VERSION}."
            ) from exc

        geometry_encoder_path = geometry_encoder_path or getattr(config, "geometry_encoder_path", None)
        if use_geometry_model:
            from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

            load_class = Qwen3_5ForConditionalGenerationWithGeometry
        else:
            load_class = Qwen3_5ForConditionalGeneration

        load_kwargs = {
            "config": config,
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }
        if use_geometry_model:
            load_kwargs["geometry_encoder_path"] = geometry_encoder_path

        if use_flash_attention_2:
            self._model = load_class.from_pretrained(pretrained, attn_implementation="flash_attention_2", **load_kwargs).eval()
        else:
            self._model = load_class.from_pretrained(pretrained, **load_kwargs).eval()

        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            padding_side="left",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        if max_length is not None:
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self._max_length = getattr(self._tokenizer, "model_max_length", None)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def uses_geometry_encoder_for_eval(self):
        return bool(getattr(self.config, "use_geometry_encoder", False))

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3.5")

    def _normalize_visual(self, visual_group):
        if isinstance(visual_group, tuple):
            visual_group = list(visual_group)
        if not isinstance(visual_group, list):
            return visual_group
        if len(visual_group) == 0:
            return None
        if len(visual_group) == 1:
            return visual_group[0]
        return visual_group

    def _sample_video_frames(self, video_path: str) -> List[Image.Image]:
        if self.use_custom_video_loader:
            return read_video_pyav_pil(
                video_path,
                num_frm=self.max_num_frames,
                fps=self.fps,
                max_image_size=self.max_image_size,
            )

        vr = decord.VideoReader(video_path)
        frame_count = len(vr)
        if frame_count <= self.max_num_frames:
            indices = np.arange(frame_count)
        else:
            indices = np.linspace(0, frame_count - 1, self.max_num_frames).astype(int)
        return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in indices]

    def _build_sample(self, context, visual):
        sample_images = []
        user_content = []

        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            frames = self._sample_video_frames(visual)
            for idx, frame in enumerate(frames):
                if self.add_frame_index:
                    user_content.append({"type": "text", "text": f"Frame-{idx}: "})
                # Keep raw PIL inputs in the chat payload to avoid per-frame base64 encoding.
                user_content.append({"type": "image", "image": frame})
                sample_images.append(frame)
        elif isinstance(visual, str) and is_image_path(visual):
            frame = Image.open(visual).convert("RGB")
            user_content.append({"type": "image", "image": frame})
            sample_images.append(frame)
        elif isinstance(visual, Image.Image):
            frame = visual.convert("RGB")
            user_content.append({"type": "image", "image": frame})
            sample_images.append(frame)
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
            for idx, frame in enumerate(visual):
                rgb = frame.convert("RGB")
                if self.add_frame_index:
                    user_content.append({"type": "text", "text": f"Frame-{idx}: "})
                user_content.append({"type": "image", "image": rgb})
                sample_images.append(rgb)
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, str) and is_image_path(v) for v in visual):
            for idx, frame_path in enumerate(visual):
                rgb = Image.open(frame_path).convert("RGB")
                if self.add_frame_index:
                    user_content.append({"type": "text", "text": f"Frame-{idx}: "})
                user_content.append({"type": "image", "image": rgb})
                sample_images.append(rgb)

        user_content.append({"type": "text", "text": context})
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]
        return message, sample_images

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            split_name = split[0]
            batched_visuals = [doc_to_visual[i](self.task_dict[task_name][split_name][ids]) for i, ids in enumerate(doc_id)]
            batch_start = time.perf_counter()

            gen_kwargs = dict(all_gen_kwargs[0])
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be Union[str, list], got {type(until)}")

            messages = []
            sample_images = []
            for context, raw_visual in zip(contexts, batched_visuals):
                visual = self._normalize_visual(raw_visual)
                message, images = self._build_sample(context, visual)
                messages.append(message)
                sample_images.append(images)

            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.disable_thinking:
                chat_template_kwargs["enable_thinking"] = False
            text = self.processor.apply_chat_template(messages, **chat_template_kwargs)
            inputs = self.processor(
                text=text,
                images=sample_images if any(len(images) > 0 for images in sample_images) else None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            if self.uses_geometry_encoder_for_eval():
                if len(sample_images) != 1:
                    raise ValueError("Qwen3.5 geometry eval currently expects per-device batch size 1.")
                geometry_encoder_inputs = build_qwen3_5_geometry_inputs(
                    sample_images[0],
                    inputs["image_grid_thw"],
                )
                inputs["geometry_encoder_inputs"] = [torch.stack(geometry_encoder_inputs)]
            preprocess_elapsed = time.perf_counter() - batch_start

            if self.device_map == "auto":
                inputs = move_qwen3_5_eval_inputs_to_device(inputs, "cuda")
            else:
                inputs = move_qwen3_5_eval_inputs_to_device(inputs, self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            generate_start = time.perf_counter()
            output_ids = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=gen_kwargs["temperature"] > 0,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            generate_elapsed = time.perf_counter() - generate_start

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            decode_elapsed = time.perf_counter() - generate_start - generate_elapsed
            input_tokens = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else -1
            output_tokens = (
                sum(int(ids.shape[-1]) for ids in generated_ids_trimmed) if generated_ids_trimmed else 0
            )
            eval_logger.debug(
                f"Qwen3.5 eval batch size={len(contexts)} input_tokens={input_tokens} "
                f"output_tokens={output_tokens} preprocess={preprocess_elapsed:.3f}s "
                f"generate={generate_elapsed:.3f}s decode={decode_elapsed:.3f}s "
                f"fast_path={self.fast_path_runtime}"
            )

            for answer, context in zip(answers, contexts):
                final_answer = strip_thinking_content(answer) if self.strip_thinking else answer
                res.append(final_answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), final_answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
