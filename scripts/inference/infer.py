#!/usr/bin/env python3

import argparse
import base64
import copy
import json
from io import BytesIO
from pathlib import Path
from typing import List

import decord
import numpy as np
import torch
from PIL import Image
from packaging.version import Version
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from qwen_vl.data.utils import load_and_preprocess_images, preprocess_pil_images

try:
    from qwen_vl_utils import extract_vision_info
except ImportError as exc:
    raise RuntimeError("qwen_vl_utils is required. Please install dependencies first.") from exc


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL_PATH = "Journey9ni/SpatialStack-Qwen2.5-4B"
MIN_QWEN3_5_TRANSFORMERS_VERSION = Version("5.3.0")
QWEN3_5_MODEL_TYPES = {"qwen3_5", "qwen3_5_vl"}


def parse_args():
    parser = argparse.ArgumentParser(description="Single-sample inference for SpatialStack models.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"HF model id or local checkpoint path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--prompt", required=True, help="User prompt text")

    visual_group = parser.add_mutually_exclusive_group(required=True)
    visual_group.add_argument("--image", type=str, help="Path to one image")
    visual_group.add_argument("--image-dir", type=str, help="Directory of images (sorted by filename)")
    visual_group.add_argument("--video", type=str, help="Path to one video")

    parser.add_argument("--device", default="cuda:0", help="Runtime device, e.g. cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-num-frames", type=int, default=32, help="Max frames for video/image-dir")
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--add-frame-index", action="store_true", help="Insert 'Frame-i:' tokens before each image")
    parser.add_argument("--no-flash-attn2", action="store_true", help="Disable flash_attention_2")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen3.5 thinking mode when supported by the chat template.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--output-json", type=str, default="", help="Optional output JSON path")
    return parser.parse_args()


def sample_indices(total: int, max_count: int) -> np.ndarray:
    if total <= max_count:
        return np.arange(total, dtype=int)
    return np.linspace(0, total - 1, max_count, dtype=int)


def load_visuals(args) -> List[Image.Image]:
    if args.image:
        return [Image.open(args.image).convert("RGB")]

    if args.image_dir:
        image_dir = Path(args.image_dir)
        files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
        if not files:
            raise ValueError(f"No image files found in {args.image_dir}")
        indices = sample_indices(len(files), args.max_num_frames)
        return [Image.open(files[i]).convert("RGB") for i in indices]

    vr = decord.VideoReader(args.video)
    indices = sample_indices(len(vr), args.max_num_frames)
    return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in indices]


def image_to_data_uri(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def decode_message_image(image) -> Image.Image:
    if isinstance(image, str) and "base64," in image:
        _, base64_data = image.split("base64,", 1)
        with BytesIO(base64.b64decode(base64_data)) as bio:
            return copy.deepcopy(Image.open(bio)).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def resolve_model_family(config) -> str:
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen2_5_vl":
        return "qwen2_5_vl"
    if model_type == "qwen2_vl":
        return "qwen2_vl"
    if model_type in QWEN3_5_MODEL_TYPES:
        return "qwen3_5"
    raise ValueError(
        f"Unsupported model_type '{model_type}' for model {getattr(config, 'name_or_path', '<unknown>')}."
    )


def require_qwen3_5_support():
    current_version = Version(transformers.__version__)
    if current_version < MIN_QWEN3_5_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Qwen3.5 inference requires transformers>="
            f"{MIN_QWEN3_5_TRANSFORMERS_VERSION}, but found {transformers.__version__}. "
            "Please upgrade dependencies before loading a Qwen3.5 checkpoint."
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
        # Qwen3.5 multimodal inputs use 3D MRoPE position ids and are not packed sequences.
        if position_ids is not None and getattr(position_ids, "ndim", None) == 3:
            return False
        return original_is_packed_sequence(position_ids, batch_size)

    flash_attention_utils._is_packed_sequence = patched_is_packed_sequence
    flash_attention_utils._spatialstack_qwen3_5_mrope_patch = True


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


def resolve_model_class(model_family: str, use_geometry_model: bool):
    if model_family == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        return (
            __import__(
                "qwen_vl.model.modeling_qwen2_5_vl",
                fromlist=["Qwen2_5_VLForConditionalGenerationWithVGGT"],
            ).Qwen2_5_VLForConditionalGenerationWithVGGT
            if use_geometry_model
            else Qwen2_5_VLForConditionalGeneration
        )

    if model_family == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration

        if use_geometry_model:
            raise NotImplementedError("Geometry-enabled inference is only implemented for Qwen2.5-VL checkpoints.")
        return Qwen2VLForConditionalGeneration

    if model_family == "qwen3_5":
        require_qwen3_5_support()
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Your transformers build does not expose Qwen3_5ForConditionalGeneration. "
                f"Please install transformers>={MIN_QWEN3_5_TRANSFORMERS_VERSION}."
            ) from exc
        if use_geometry_model:
            from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

            return Qwen3_5ForConditionalGenerationWithGeometry
        return Qwen3_5ForConditionalGeneration

    raise ValueError(f"Unsupported model family: {model_family}")


def build_messages(prompt: str, visuals: List[Image.Image], add_frame_index: bool):
    content = []
    for idx, img in enumerate(visuals):
        if add_frame_index:
            content.append({"type": "text", "text": f"Frame-{idx}: "})
        content.append({"type": "image", "image": image_to_data_uri(img)})
    content.append({"type": "text", "text": prompt})

    return [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]]


def prepare_visual_inputs(messages, processor):
    image_inputs = []
    geometry_encoder_inputs = []

    patch_size = processor.image_processor.patch_size
    merge_size = processor.image_processor.merge_size

    for message in messages:
        vision_info = extract_vision_info(message)
        cur_geo_inputs = []

        for ele in vision_info:
            if "image" not in ele:
                continue
            image = decode_message_image(ele["image"])

            image = load_and_preprocess_images([image])[0]
            cur_geo_inputs.append(copy.deepcopy(image))

            _, height, width = image.shape
            if (width // patch_size) % merge_size > 0:
                width -= (width // patch_size) % merge_size * patch_size
            if (height // patch_size) % merge_size > 0:
                height -= (height // patch_size) % merge_size * patch_size

            image_inputs.append(image[:, :height, :width])

        geometry_encoder_inputs.append(torch.stack(cur_geo_inputs))

    return image_inputs, geometry_encoder_inputs


def prepare_raw_visual_inputs(messages):
    image_inputs = []
    for message in messages:
        vision_info = extract_vision_info(message)
        for ele in vision_info:
            if "image" not in ele:
                continue
            image_inputs.append(decode_message_image(ele["image"]))
    return image_inputs


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({args.device}) but no CUDA is available.")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    config = AutoConfig.from_pretrained(args.model_path)
    model_family = resolve_model_family(config)
    use_geometry_model = getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False)
    model_class = resolve_model_class(model_family, use_geometry_model)
    if model_family == "qwen3_5":
        patch_qwen3_5_flash_attention()

    model_kwargs = {
        "pretrained_model_name_or_path": args.model_path,
        "config": config,
        "torch_dtype": torch_dtype,
        "device_map": args.device,
    }
    if model_family == "qwen3_5" and use_geometry_model:
        geometry_encoder_path = getattr(config, "geometry_encoder_path", None)
        if geometry_encoder_path:
            model_kwargs["geometry_encoder_path"] = geometry_encoder_path

    if args.no_flash_attn2:
        model = model_class.from_pretrained(**model_kwargs).eval()
    else:
        model = model_class.from_pretrained(**model_kwargs, attn_implementation="flash_attention_2").eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        padding_side="left",
    )

    visuals = load_visuals(args)
    messages = build_messages(args.prompt, visuals, args.add_frame_index)
    chat_template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_family == "qwen3_5" and args.disable_thinking:
        chat_template_kwargs["enable_thinking"] = False
    text = processor.apply_chat_template(messages, **chat_template_kwargs)
    if model_family == "qwen3_5":
        image_inputs = prepare_raw_visual_inputs(messages)
        geometry_encoder_inputs = None
        if use_geometry_model:
            image_inputs = preprocess_pil_images(image_inputs, mode="crop", target_size=518)
        model_inputs = processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        if use_geometry_model:
            geometry_encoder_inputs = [torch.stack(build_qwen3_5_geometry_inputs(image_inputs, model_inputs["image_grid_thw"]))]
    else:
        image_inputs, geometry_encoder_inputs = prepare_visual_inputs(messages, processor)
        model_inputs = processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False,
        )

    if use_geometry_model:
        model_inputs["geometry_encoder_inputs"] = [feat.to(args.device) for feat in geometry_encoder_inputs]
    model_inputs = model_inputs.to(args.device)

    output_ids = model.generate(
        **model_inputs,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, output_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    result = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "num_visuals": len(visuals),
        "response": answer,
    }
    print(answer)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
