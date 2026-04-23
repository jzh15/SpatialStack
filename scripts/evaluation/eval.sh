#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
fi

cd "$PROJECT_ROOT"

QWEN35_ENV_ROOT="${QWEN35_ENV_ROOT:-$HOME/.conda/envs/spatialstack-qwen35}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

if [[ ! -d "$CUDA_HOME" ]]; then
    echo "[ERROR] Missing CUDA toolkit directory: $CUDA_HOME"
    exit 1
fi

runtime_path="${PATH:-/usr/local/bin:/usr/bin:/bin}"
if [[ -d "$QWEN35_ENV_ROOT/bin" ]]; then
    runtime_path="$QWEN35_ENV_ROOT/bin:$runtime_path"
fi
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/cache}"

export CUDA_HOME
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export PATH="$CUDA_HOME/bin:$runtime_path"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$CACHE_ROOT/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp/$USER}/spatialstack-triton/${SLURM_JOB_ID:-manual}/${SLURM_PROCID:-0}}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"
mkdir -p "$TRITON_CACHE_DIR"

LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"
export LMMS_EVAL_LAUNCHER
export NCCL_NVLS_ENABLE=0

# Allow overriding the key parameters via environment variables.
DEFAULT_BENCHMARKS="cvbench, blink_spatial, sparbench, videomme, mmsibench"

BENCHMARKS_RAW="${BENCHMARKS:-}"
if [[ -z "$BENCHMARKS_RAW" ]]; then
    BENCHMARKS_RAW="${BENCHMARK:-$DEFAULT_BENCHMARKS}" # choices: [vsibench, cvbench, blink_spatial, sparbench, videomme, mmsibench]
fi
IFS=',' read -ra BENCHMARK_LIST <<< "$BENCHMARKS_RAW"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
MODEL_IMPL="${MODEL_IMPL:-spatialstack}"
MODEL_ARGS_BASE="${MODEL_ARGS_BASE:-pretrained=$MODEL_PATH,use_flash_attention_2=true,max_num_frames=32,max_length=12800}"
MODEL_ARGS_EXTRA="${MODEL_ARGS_EXTRA:-}"
GEN_KWARGS="${GEN_KWARGS:-}"
LIMIT="${LIMIT:-}"
VERBOSITY="${VERBOSITY:-INFO}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/eval}"
TIMESTAMP="$(date "+%Y%m%d")"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/${TIMESTAMP}}"

mkdir -p "$OUTPUT_PATH"
echo "[INFO] cuda_home=$CUDA_HOME"
echo "[INFO] hf_home=$HF_HOME"
echo "[INFO] hf_hub_cache=$HUGGINGFACE_HUB_CACHE"
echo "[INFO] hf_datasets_cache=$HF_DATASETS_CACHE"
echo "[INFO] triton_cache_dir=$TRITON_CACHE_DIR"

NUM_MACHINES="${NUM_MACHINES:-}"
if [[ -z "$NUM_MACHINES" ]]; then
    if [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
        NUM_MACHINES="${SLURM_JOB_NUM_NODES}"
    elif [[ -n "${SLURM_NNODES:-}" ]]; then
        NUM_MACHINES="${SLURM_NNODES}"
    else
        NUM_MACHINES=1
    fi
fi

PROCESSES_PER_MACHINE="${PROCESSES_PER_MACHINE:-}"
if [[ -z "$PROCESSES_PER_MACHINE" ]]; then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" -gt 0 ]]; then
        PROCESSES_PER_MACHINE="${SLURM_GPUS_ON_NODE}"
    elif [[ -n "${SLURM_TASKS_PER_NODE:-}" ]]; then
        PROCESSES_PER_MACHINE="${SLURM_TASKS_PER_NODE%%(*}"
    else
        if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            IFS=',' read -ra __cvd <<< "${CUDA_VISIBLE_DEVICES}"
            if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
                PROCESSES_PER_MACHINE=0
            else
                PROCESSES_PER_MACHINE="${#__cvd[@]}"
            fi
        elif command -v nvidia-smi >/dev/null 2>&1; then
            PROCESSES_PER_MACHINE="$(nvidia-smi -L | wc -l | tr -d ' ')"
        else
            PROCESSES_PER_MACHINE="$(python - <<'PY'
import os
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
        fi
    fi
fi

if [[ -z "$PROCESSES_PER_MACHINE" || "$PROCESSES_PER_MACHINE" -lt 1 ]]; then
    PROCESSES_PER_MACHINE=1
fi

if [[ -z "$NUM_MACHINES" || "$NUM_MACHINES" -lt 1 ]]; then
    NUM_MACHINES=1
fi

TOTAL_PROCESSES=$((NUM_MACHINES * PROCESSES_PER_MACHINE))
if [[ "$TOTAL_PROCESSES" -lt 1 ]]; then
    TOTAL_PROCESSES=1
fi

MACHINE_RANK="${MACHINE_RANK:-}"
if [[ -z "$MACHINE_RANK" ]]; then
    if [[ -n "${SLURM_PROCID:-}" ]]; then
        MACHINE_RANK="${SLURM_PROCID}"
    elif [[ -n "${SLURM_NODEID:-}" ]]; then
        MACHINE_RANK="${SLURM_NODEID}"
    else
        MACHINE_RANK=0
    fi
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export MASTER_PORT="${MASTER_PORT:-29500}"
    if [[ -z "${MASTER_ADDR:-}" ]]; then
        if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_NODELIST:-}" ]]; then
            export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
        else
            export MASTER_ADDR="$(hostname)"
        fi
    fi
fi

accelerate_args=(--num_processes "$TOTAL_PROCESSES")
if [[ "$NUM_MACHINES" -gt 1 ]]; then
    accelerate_args+=(--num_machines "$NUM_MACHINES" --machine_rank "$MACHINE_RANK")
fi
if [[ "$TOTAL_PROCESSES" -gt 1 ]]; then
    accelerate_args+=(--multi_gpu)
fi
if [[ -n "${MASTER_PORT:-}" ]]; then
    accelerate_args+=(--main_process_port "$MASTER_PORT")
fi
if [[ -n "${MASTER_ADDR:-}" ]]; then
    accelerate_args+=(--main_process_ip "$MASTER_ADDR")
fi

torchrun_args=()
if [[ "$TOTAL_PROCESSES" -gt 1 ]]; then
    torchrun_args+=(
        --nproc_per_node "$PROCESSES_PER_MACHINE"
        --nnodes "$NUM_MACHINES"
        --node_rank "$MACHINE_RANK"
        --master_addr "${MASTER_ADDR:-127.0.0.1}"
        --master_port "${MASTER_PORT:-29500}"
    )
fi

model_args="$MODEL_ARGS_BASE"
if [[ "$MODEL_ARGS_BASE" != *"pretrained="* ]]; then
    model_args="pretrained=$MODEL_PATH,$model_args"
fi
if [[ -n "$MODEL_ARGS_EXTRA" ]]; then
    model_args="$model_args,$MODEL_ARGS_EXTRA"
fi
gen_kwargs_args=()
if [[ -n "$GEN_KWARGS" ]]; then
    gen_kwargs_args=(--gen_kwargs "$GEN_KWARGS")
fi
limit_args=()
if [[ -n "$LIMIT" ]]; then
    limit_args=(--limit "$LIMIT")
fi

for raw_benchmark in "${BENCHMARK_LIST[@]}"; do
    benchmark="${raw_benchmark//[[:space:]]/}"
    if [[ -z "$benchmark" ]]; then
        continue
    fi
    task_name="$benchmark"
    task_output="$OUTPUT_PATH/$benchmark"
    mkdir -p "$task_output"

    eval_args=(
        --model "$MODEL_IMPL"
        --model_args "$model_args"
        --tasks "$task_name"
        --batch_size 1
        --output_path "$task_output"
        "${gen_kwargs_args[@]}"
        "${limit_args[@]}"
        --verbosity "$VERBOSITY"
        --log_samples
    )

    if [[ "$LMMS_EVAL_LAUNCHER" == "torchrun" ]]; then
        if [[ "$TOTAL_PROCESSES" -gt 1 ]]; then
            torchrun "${torchrun_args[@]}" -m lmms_eval "${eval_args[@]}"
        else
            python -m lmms_eval "${eval_args[@]}"
        fi
    else
        accelerate launch "${accelerate_args[@]}" -m lmms_eval "${eval_args[@]}"
    fi
done
