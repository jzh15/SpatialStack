# Training

This guide covers the Qwen3.5 training workflow used by `scripts/train/train.sh`.
Run all commands below from the repository root after completing environment
setup in `README.md`.

## Data Preparation

The default training mix reads these paths:

- `data/train/spar_234k.json`
- `data/train/llava_hound_64k.json`
- `data/vlm3r/annotations/vsibench_train/merged_qa_scannet_train.json`
- `data/vsi_590k/annotations/vsi_appearance_order_vsibench_scannet.json`

Download annotations and map them to the paths above.
The `Journey9ni/SpatialStackData` dataset now stores annotation JSON files at the
repository root. The media payload has been removed from that dataset repo, so
only annotations should be downloaded from it.

```bash
mkdir -p ./data/annotations

hf download Journey9ni/SpatialStackData \
  --repo-type dataset \
  --include "*.json" \
  --local-dir ./data/annotations

mkdir -p ./data/train
mkdir -p ./data/vlm3r/annotations/vsibench_train
mkdir -p ./data/vsi_590k/annotations

ln -sfn ../annotations/spar_234k.json \
  ./data/train/spar_234k.json
ln -sfn ../annotations/llava_hound_64k.json \
  ./data/train/llava_hound_64k.json
ln -sfn ../../../annotations/merged_qa_scannet_train.json \
  ./data/vlm3r/annotations/vsibench_train/merged_qa_scannet_train.json
ln -sfn ../../annotations/vsi_appearance_order_vsibench_scannet.json \
  ./data/vsi_590k/annotations/vsi_appearance_order_vsibench_scannet.json
```

Download the media used by the same default mix.

SPAR:

The `SPAR-7M` download is published as split chunks of one large
`tar.gz` archive. The files named `spar-00.tar.gz`, `spar-01.tar.gz`, ... are
not individually extractable; concatenate them in order and stream the combined
archive into `tar`.

```bash
mkdir -p ./data/media/spar

hf download jasonzhango/SPAR-7M \
  --repo-type dataset \
  --revision 976c19177468eabe64e9e2dd0f0450cd32dacc1f \
  --include "spar-*.tar.gz" \
  --local-dir ./data/media/spar

(
  cd ./data/media
  cat \
    spar/spar-00.tar.gz spar/spar-01.tar.gz spar/spar-02.tar.gz spar/spar-03.tar.gz \
    spar/spar-04.tar.gz spar/spar-05.tar.gz spar/spar-06.tar.gz spar/spar-07.tar.gz \
    spar/spar-08.tar.gz spar/spar-09.tar.gz spar/spar-10.tar.gz spar/spar-11.tar.gz \
    spar/spar-12.tar.gz spar/spar-13.tar.gz \
  | pigz -dc | tar -xf - -C ./.
)
```

After extraction, the training paths should exist directly under `./data/media/spar/`,
for example `./data/media/spar/scannet/...` and `./data/media/spar/structured3d/...`.
If you accidentally extracted inside `./data/media/spar/`, move `./data/media/spar/spar/*`
up one level before training.

LLaVA-Hound:

```bash
mkdir -p ./data/media/llava_hound

hf download ShareGPTVideo/train_video_and_instruction \
  --repo-type dataset \
  --include "train_300k/**" \
  --local-dir ./data/media/llava_hound

mkdir -p ./data/media/llava_hound/frames
find ./data/media/llava_hound/train_300k -maxdepth 1 -name 'chunk_*.tar.gz' -print0 \
| xargs -0 -P"$(nproc)" -I{} tar -I pigz -x -f "{}" -C ./data/media/llava_hound/frames
```

VLM-3R ScanNet video:

```bash
mkdir -p ./data/vlm3r/media/scannet

hf download Journey9ni/aweb \
  --repo-type dataset \
  --include "ScanNet/videos/train/**" \
  --local-dir ./data/vlm3r/media/scannet

mv ./data/vlm3r/media/scannet/ScanNet/videos/train \
  ./data/vlm3r/media/scannet/videos
```

VSI-590K reuses the same ScanNet videos:

```bash
mkdir -p ./data/vsi_590k/media
ln -sfn ../../vlm3r/media/scannet/videos ./data/vsi_590k/media/scannet
```

If your shared project directory is close to its inode limit, place
high-file-count media trees such as `SPAR` or `llava_hound/frames` under your
personal scratch and symlink them back into `./data/media/...`.

## Launch Training

Use the unified training script:

```bash
bash scripts/train/train.sh
```

You can either edit `scripts/train/train.sh` directly or override the main parameters with environment variables.

Before running, set these parameters in `scripts/train/train.sh` or via env vars:

- `MODEL_PATH`: base VLM path or HF id, typically `Qwen/Qwen3.5-4B`
- `OUTPUT_DIR`: checkpoint/log directory (default: `./output/spatialstack_train`)
- `CACHE_DIR`: model cache directory (default: `./cache`)
- `DATASETS`: training datasets and sampling ratio string  
  default: `spar_234k%60,llava_hound_64k%60,vlm3r_scannet%60,vsi_appr_order%50`
- `LR`: learning rate (default: `1e-5`)
- `TOTAL_BATCH_SIZE`: global batch size used to compute `gradient_accumulation_steps`
- `USE_GEOMETRY_ENCODER`: keep this `False` for the documented Qwen3.5 workflow
- `DATA_FLATTEN`: keep this `False` for the documented Qwen3.5 workflow
- `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NODE_RANK`, `CUDA_VISIBLE_DEVICES`: distributed launch controls (optional)

### Qwen3.5 training

The public branch documents only the official Qwen3.5 base-model training path.

Use the Python 3.12 Qwen3.5 environment from [README.md](./README.md), then launch:

```bash
MODEL_PATH=Qwen/Qwen3.5-4B \
USE_GEOMETRY_ENCODER=False \
DATA_FLATTEN=False \
OUTPUT_DIR=./output/qwen35_stock_train \
bash scripts/train/train.sh
```

Qwen3.5 notes:

- Keep `USE_GEOMETRY_ENCODER=False`; geometry training is not part of the documented public Qwen3.5 workflow.
- Keep `DATA_FLATTEN=False`; the packed-sequence path is not part of the documented public Qwen3.5 workflow.
- The saved checkpoint is intended to stay compatible with the existing `infer.py` and `lmms_eval --model qwen3_5` paths in this repository.
- For multi-node launches, prefer a local model snapshot path over the raw HF id. We observed more reliable startup on large jobs when `MODEL_PATH` points at a pre-downloaded snapshot.

#### Example 64-GPU Slurm launch

A reference Slurm batch script is provided for multi-node training (8 nodes × 8 GPUs = 64 GPUs):

```bash
sbatch scripts/train/slurm/run_qwen35_64gpu_vision.sbatch
```

Before submission, edit the batch script to set your cluster's partition, account, and conda environment path. Key environment variable overrides:

```bash
MODEL_PATH=/path/to/local/qwen35_snapshot \
OUTPUT_DIR=./output/my_run \
DATASETS=llava_hound_64k%1 \
TOTAL_BATCH_SIZE=64 \
sbatch scripts/train/slurm/run_qwen35_64gpu_vision.sbatch
```
