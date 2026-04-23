#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation
set -euo pipefail

# ======================
# Distributed Configuration
# ======================
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}               # [Required] Master node IP for multi-GPU training
MASTER_PORT=${MASTER_PORT:-22223}                   # Default rendezvous port

# ======================
# Slurm auto-configuration (overrides defaults when available)
# ======================
if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
fi

NODE_RANK=${NODE_RANK:-${SLURM_PROCID:-0}}
NNODES=${NNODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}}

# Prefer CUDA_VISIBLE_DEVICES to honor manual GPU selection before fallback detection
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VIS_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
    IFS=',' read -r -a __CUDA_DEVICE_LIST <<< "$CUDA_VIS_DEVICES"
    NPROC_PER_NODE=0
    for __dev in "${__CUDA_DEVICE_LIST[@]}"; do
        if [[ -z "$__dev" ]]; then
            continue
        elif [[ "$__dev" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            __start=${BASH_REMATCH[1]}
            __end=${BASH_REMATCH[2]}
            if (( __end >= __start )); then
                NPROC_PER_NODE=$((NPROC_PER_NODE + __end - __start + 1))
            fi
        else
            NPROC_PER_NODE=$((NPROC_PER_NODE + 1))
        fi
    done
fi

if [ -z "${NPROC_PER_NODE:-}" ] || [ "$NPROC_PER_NODE" -le 0 ]; then
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
    else
        NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
    fi
fi

if [ "$NPROC_PER_NODE" -le 0 ]; then
    echo ">>>>> No visible GPUs detected; defaulting to 1 process"
    NPROC_PER_NODE=1
fi

if [ -z "${NNODES:-}" ] || [ "$NNODES" -le 0 ]; then
    NNODES=1
fi

# WORLD_SIZE is used to compute gradient accumulation; helpful to export for torchrun as well
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
export WORLD_SIZE
export NODE_RANK

# ======================
# Path Configuration
# ======================

# For local runs you may switch these to HF ids, e.g.:
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-1B}"

###################################### ENV DIVIDER
OUTPUT_DIR="${OUTPUT_DIR:-./output/spatialstack_train}"              # Directory for saving checkpoints
CACHE_DIR="${CACHE_DIR:-./cache}"                                    # [TrainingArguments] Cache directory for models
mkdir -p "$OUTPUT_DIR"

# ======================
# Training Hyperparameters
# ======================
LR="${LR:-1e-5}"
total_batch_size="${TOTAL_BATCH_SIZE:-64}"

if [ "$WORLD_SIZE" -gt 0 ]; then
    GRADIENT_ACCUMULATION_STEPS=$(( total_batch_size / WORLD_SIZE ))
else
    GRADIENT_ACCUMULATION_STEPS=$total_batch_size
fi
if [ "$GRADIENT_ACCUMULATION_STEPS" -le 0 ]; then
    echo ">>>>> gradient accumulation would be <1; forcing to 1 (total_batch_size=$total_batch_size, world_size=$WORLD_SIZE)"
    GRADIENT_ACCUMULATION_STEPS=1
fi
echo ">>>>> grad accum = $GRADIENT_ACCUMULATION_STEPS"

# ======================
# Model Configuration
# ======================
DATASETS="${DATASETS:-spar_234k%60,llava_hound_64k%60,vlm3r_scannet%60,vsi_appr_order%50}"             # [DataArguments] Dataset list
GEOMETRY_ENCODER_TYPE="${GEOMETRY_ENCODER_TYPE:-vggt}"
USE_GEOMETRY_ENCODER="${USE_GEOMETRY_ENCODER:-true}"
DATA_FLATTEN="${DATA_FLATTEN:-False}"
FEATURE_FUSION_METHOD="${FEATURE_FUSION_METHOD:-deepstack_language_add}"
GEOMETRY_FUSION_LAYERS="${GEOMETRY_FUSION_LAYERS:-0 1 2}"
GEOMETRY_ENCODER_LAYERS="${GEOMETRY_ENCODER_LAYERS:-11 17 23}"
VISION_LANGUAGE_FUSION_LAYERS="${VISION_LANGUAGE_FUSION_LAYERS:-}"

train_args=(
         --model_name_or_path "$MODEL_PATH"
         --tune_mm_llm True
         --tune_mm_vision False
         --tune_mm_mlp False
         --dataset_use "$DATASETS"
         --output_dir "$OUTPUT_DIR"
         --cache_dir "$CACHE_DIR"
         --bf16
         --per_device_train_batch_size 1
         --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
         --learning_rate "$LR"
         --mm_projector_lr 1e-5
         --vision_tower_lr 1e-6
         --optim adamw_torch
         --model_max_length 12800
         --data_flatten "$DATA_FLATTEN"
         --max_pixels $((576*28*28))
         --min_pixels $((16*28*28))
         --base_interval 2
         --video_max_frames 8
         --video_min_frames 4
         --video_max_frame_pixels $((1664*28*28))
         --video_min_frame_pixels $((256*28*28))
         --num_train_epochs 1
         --warmup_ratio 0.03
         --lr_scheduler_type cosine
         --weight_decay 0.01
         --logging_steps 10
         --save_steps 1000
         --save_total_limit 10
         --deepspeed scripts/zero2_opt.json
         --gradient_checkpointing
         --dataloader_num_workers 4
         --group_by_modality_length true
         --seed 0
         --report_to none
         --use_geometry_encoder "$USE_GEOMETRY_ENCODER"
)

if [[ "${USE_GEOMETRY_ENCODER,,}" == "true" ]]; then
    train_args+=(
         --geometry_encoder_type "$GEOMETRY_ENCODER_TYPE"
         --geometry_encoder_path "$GEOMETRY_ENCODER_PATH"
         --feature_fusion_method "$FEATURE_FUSION_METHOD"
         --geometry_fusion_layers ${GEOMETRY_FUSION_LAYERS}
         --geometry_encoder_layers ${GEOMETRY_ENCODER_LAYERS}
    )
    if [[ -n "${VISION_LANGUAGE_FUSION_LAYERS}" ]]; then
        train_args+=(
             --vision_language_fusion_layers ${VISION_LANGUAGE_FUSION_LAYERS}
        )
    fi
fi

torchrun --nproc_per_node=$NPROC_PER_NODE \
         --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         src/qwen_vl/train/train_qwen.py \
         "${train_args[@]}" \
         2>&1 | tee ${OUTPUT_DIR}/train.rank${NODE_RANK}.log
