#!/bin/bash
# Set your cluster account below if your scheduler requires it.
# #SBATCH --account=<your-account>
#SBATCH --partition=gpu            # Change to your cluster's GPU partition
#SBATCH --job-name=q35-vsi-mn
#SBATCH --nodes=12
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
fi

cd "$PROJECT_ROOT"
mkdir -p logs/slurm
mkdir -p cache/triton

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/output/qwen35_64gpu_geometry_fix_20260321_222108}"
GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-1B}"
QWEN35_ENV_ROOT="${QWEN35_ENV_ROOT:-$HOME/.conda/envs/spatialstack-qwen35}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/eval/qwen35_64gpu_geometry_fix_20260321_222108_vsibench_48gpu_short}"
BENCHMARKS="${BENCHMARKS:-vsibench}"
MODEL_IMPL="${MODEL_IMPL:-qwen3_5}"
MODEL_ARGS_BASE="${MODEL_ARGS_BASE:-pretrained=$MODEL_PATH,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=$GEOMETRY_ENCODER_PATH,disable_thinking=true}"

export PROJECT_ROOT
export MODEL_PATH
export GEOMETRY_ENCODER_PATH
export QWEN35_ENV_ROOT
export OUTPUT_ROOT
export BENCHMARKS
export MODEL_IMPL
export MODEL_ARGS_BASE

srun --ntasks="${SLURM_JOB_NUM_NODES:-12}" --ntasks-per-node=1 bash -lc '
    set -euo pipefail
    cd "'"$PROJECT_ROOT"'"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export QWEN35_ENV_ROOT="'"$QWEN35_ENV_ROOT"'"
    export MODEL_PATH="'"$MODEL_PATH"'"
    export GEOMETRY_ENCODER_PATH="'"$GEOMETRY_ENCODER_PATH"'"
    export OUTPUT_ROOT="'"$OUTPUT_ROOT"'"
    export BENCHMARKS="'"$BENCHMARKS"'"
    export MODEL_IMPL="'"$MODEL_IMPL"'"
    export MODEL_ARGS_BASE="'"$MODEL_ARGS_BASE"'"
    export TRITON_CACHE_DIR="'"$PROJECT_ROOT"'/cache/triton/${SLURM_JOB_ID:-manual}/${SLURM_PROCID:-0}"
    bash scripts/evaluation/eval.sh
'
