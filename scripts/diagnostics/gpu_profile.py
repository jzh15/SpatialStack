#!/usr/bin/env python3
"""Locate the bottleneck when SpatialStack evaluation runs far slower than expected.

The probes run from cheapest to most specific, so the first one that deviates
from the reference points at the cause:

  [1] bf16 GEMM            - is cuBLAS using native kernels for this GPU arch?
  [2] SDPA at VGGT's shape - does PyTorch pick a flash backend for the 32-frame
                             global attention (~38.6k tokens), or fall back to
                             the math path (100x slower, tens of GB)?
  [3] VGGT geometry encoder- the dominant cost of a SpatialStack eval question
  [4] Qwen3.5 LLM prefill  - the remaining cost of a question (~9.7k tokens)

Reference numbers, 1x H200, torch 2.10.0+cu129, flash_attn 2.8.3:

  [1] bf16 matmul 8192^3       :    1.4 ms  (807 TFLOP/s)
  [2] SDPA default (N=38600)   :   19.6 ms   (FLASH 19.5 ms / MEM_EFF 45.0 ms)
  [3] VGGT encoder (32 frames) :  885.0 ms
  [4] LLM prefill 9.7k tokens  :  197.7 ms

Those add up to the ~1.24 s per question that VSI-Bench evaluation spends on
the GPU (plus ~1.0 s of CPU-side video decoding), i.e. ~2.3 s per question and
~3.2 h for the full 5130-question benchmark on a single GPU.

Usage:

    python scripts/diagnostics/gpu_profile.py                    # all probes
    python scripts/diagnostics/gpu_profile.py --skip-model       # [1] and [2] only
"""

import argparse
import time

import torch
import torch.nn.functional as F

DEFAULT_MODEL_PATH = "Journey9ni/SpatialStack-Qwen3.5-4B"
DEFAULT_GEOMETRY_ENCODER_PATH = "facebook/VGGT-1B"

# A 640x480 video frame becomes a 30x40 patch grid at eval settings, so VGGT's
# global attention sees 32 * 1200 patches plus per-frame special tokens.
VGGT_GLOBAL_SEQ_LEN = 38600
VGGT_FRAME_HEIGHT = 420
VGGT_FRAME_WIDTH = 560
PREFILL_TOKENS = 9700


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the SpatialStack evaluation forward pass.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"HF model id or local checkpoint path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--geometry-encoder-path", default=DEFAULT_GEOMETRY_ENCODER_PATH)
    parser.add_argument("--frames", type=int, default=32, help="Frames per question (default: 32)")
    parser.add_argument("--skip-model", action="store_true", help="Run only the synthetic probes [1] and [2]")
    parser.add_argument("--no-flash-attn2", action="store_true", help="Disable flash_attention_2 for probe [4]")
    return parser.parse_args()


def bench(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n


def report_environment():
    print("=" * 70)
    print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
    print(f"arch_list: {torch.cuda.get_arch_list()}")
    capability = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)} | capability sm_{capability[0]}{capability[1]}")
    try:
        import flash_attn

        print(f"flash_attn: {flash_attn.__version__}")
    except ImportError:
        print("flash_attn: NOT AVAILABLE")
    print("=" * 70)
    if f"sm_{capability[0]}{capability[1]}" not in torch.cuda.get_arch_list():
        print(
            f"[WARN] This PyTorch build has no sm_{capability[0]}{capability[1]} kernels for your GPU; "
            "expect large slowdowns. Reinstall torch as described in the README."
        )


def probe_gemm():
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    elapsed = bench(lambda: a @ b)
    print(f"[1] bf16 matmul 8192^3      : {elapsed * 1e3:7.1f} ms  ({2 * 8192 ** 3 / elapsed / 1e12:.0f} TFLOP/s)")
    del a, b
    torch.cuda.empty_cache()


def probe_sdpa(seq_len):
    q = torch.randn(1, 16, seq_len, 64, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    elapsed = bench(lambda: F.scaled_dot_product_attention(q, k, v), n=3, warmup=1)
    print(f"[2] SDPA default (N={seq_len})  : {elapsed * 1e3:7.1f} ms")

    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = (("FLASH  ", SDPBackend.FLASH_ATTENTION), ("MEM_EFF", SDPBackend.EFFICIENT_ATTENTION))
    for name, backend in backends:
        try:
            with sdpa_kernel(backend):
                elapsed = bench(lambda: F.scaled_dot_product_attention(q, k, v), n=3, warmup=1)
            print(f"      {name}              : {elapsed * 1e3:7.1f} ms")
        except Exception as exc:  # noqa: BLE001 - any backend rejection is the signal we want
            print(f"      {name}              : UNAVAILABLE ({type(exc).__name__}: {str(exc)[:70]})")
    del q, k, v
    torch.cuda.empty_cache()


def load_model(args):
    from transformers import AutoConfig

    from lmms_eval.models.qwen3_5 import patch_qwen3_5_flash_attention
    from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

    patch_qwen3_5_flash_attention()
    config = AutoConfig.from_pretrained(args.model_path)
    load_kwargs = {
        "config": config,
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda:0",
        "geometry_encoder_path": args.geometry_encoder_path,
    }
    if not args.no_flash_attn2:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    start = time.perf_counter()
    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(args.model_path, **load_kwargs).eval()
    print(f"[--] model loaded in {time.perf_counter() - start:.1f}s")
    return model, config


def probe_geometry_encoder(model, config, frames):
    encoder = getattr(model.model, "geometry_encoder", None)
    if encoder is None:
        print("[3] VGGT encoder            : SKIPPED (checkpoint has no geometry encoder)")
        return
    images = torch.rand(frames, 3, VGGT_FRAME_HEIGHT, VGGT_FRAME_WIDTH, device="cuda", dtype=torch.float32)
    layers = getattr(config, "geometry_encoder_layers", None) or [-2]
    elapsed = bench(
        lambda: encoder.encode_layers(images, layer_indices=layers, spatial_merge_size=2),
        n=2,
        warmup=1,
    )
    print(f"[3] VGGT encoder ({frames} frames): {elapsed * 1e3:7.1f} ms")
    del images
    torch.cuda.empty_cache()


def probe_prefill(model):
    input_ids = torch.randint(1000, 5000, (1, PREFILL_TOKENS), device="cuda")
    with torch.no_grad():
        elapsed = bench(lambda: model(input_ids=input_ids), n=3, warmup=1)
    print(f"[4] LLM prefill {PREFILL_TOKENS // 100 / 10}k tokens : {elapsed * 1e3:7.1f} ms")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this profile needs a GPU.")

    report_environment()
    probe_gemm()
    probe_sdpa(VGGT_GLOBAL_SEQ_LEN)
    if args.skip_model:
        return
    model, config = load_model(args)
    probe_geometry_encoder(model, config, args.frames)
    probe_prefill(model)
    print("=" * 70)


if __name__ == "__main__":
    main()
