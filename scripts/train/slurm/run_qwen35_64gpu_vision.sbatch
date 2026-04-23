#!/bin/bash
#SBATCH --job-name=qwen35-64gpu-vision
#SBATCH --partition=gpu            # Change to your cluster's GPU partition
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

# Set your cluster account with `#SBATCH --account=...` if your scheduler requires it.

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
fi

cd "$PROJECT_ROOT"
mkdir -p slurm_logs
mkdir -p .cache/triton

module load Anaconda3/2025.06-1

QWEN35_ENV_ROOT="${QWEN35_ENV_ROOT:-$HOME/.conda/envs/spatialstack-qwen35}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$QWEN35_ENV_ROOT/bin:$CUDA_HOME/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export MASTER_PORT="${MASTER_PORT:-29517}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp/$USER}/spatialstack-triton/${SLURM_JOB_ID:-manual}/${SLURM_PROCID:-0}}"
mkdir -p "$TRITON_CACHE_DIR"

# Use a repo-local snapshot by default; override with a local mirror if needed.
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
export USE_GEOMETRY_ENCODER="${USE_GEOMETRY_ENCODER:-False}"
export DATA_FLATTEN="${DATA_FLATTEN:-False}"
export OUTPUT_DIR="${OUTPUT_DIR:-./output/qwen35_64gpu_vision_${SLURM_JOB_ID}}"
export CACHE_DIR="${CACHE_DIR:-./cache}"
export LR="${LR:-1e-5}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-64}"
export DATASETS="${DATASETS:-spar_234k%60,llava_hound_64k%60,vlm3r_scannet%60,vsi_appr_order%50}"

if [[ "${DATA_FLATTEN,,}" != "false" ]]; then
    echo "[ERROR] Qwen3.5 training in this branch requires DATA_FLATTEN=False"
    exit 1
fi

"$QWEN35_ENV_ROOT/bin/python" - <<'PY'
import cv2
import decord
import deepspeed
import torch
import transformers

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("deepspeed", deepspeed.__version__)
print("decord", decord.__version__)
print("opencv", cv2.__version__)
print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
PY

echo "[INFO] project_root=$PROJECT_ROOT"
echo "[INFO] job_id=${SLURM_JOB_ID:-none} nodes=${SLURM_JOB_NUM_NODES:-8} ntasks=${SLURM_NTASKS:-8}"
echo "[INFO] model_path=$MODEL_PATH"
echo "[INFO] output_dir=$OUTPUT_DIR"
echo "[INFO] cache_dir=$CACHE_DIR"
echo "[INFO] total_batch_size=$TOTAL_BATCH_SIZE"
echo "[INFO] datasets=$DATASETS"
echo "[INFO] triton_cache_dir=$TRITON_CACHE_DIR"
echo "[INFO] cuda_home=$CUDA_HOME"
echo "[INFO] qwen35_env_root=$QWEN35_ENV_ROOT"

srun --ntasks="${SLURM_JOB_NUM_NODES:-8}" --ntasks-per-node=1 bash scripts/train/train.sh
