#!/usr/bin/env bash
set -e

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  cat <<'EOF'
Usage: ./run.sh [model] [dataset] [save_dir] [epochs] [batch_size] [device] [ddp] [nproc_per_node] [extra main.py args...]

Arguments:
  model           Model name (run `python main.py --mode list` for every registered model)
  dataset         Dataset name (cifar10, cifar100, imagenet)
  save_dir        Directory to save checkpoints
  epochs          Number of training epochs
  batch_size      Batch size for training
  device          Device string for PyTorch (cpu, cuda, cuda:0, ...)
  ddp             Set to 'ddp' or 'true' to launch distributed training with torchrun
  nproc_per_node  Number of processes / GPUs for distributed training

Anything after the 8th argument is forwarded to main.py.

Example:
  ./run.sh mobilevit cifar10 ./checkpoints 20 64 cpu
  ./run.sh vit_tiny cifar10 ./checkpoints 100 128 cuda false 1 --optimizer adamw --lr 1e-3 --warmup-epochs 5 --amp
EOF
  exit 0
fi

MODEL="${1:-mobilevit}"
DATASET="${2:-cifar10}"
SAVE_DIR="${3:-./checkpoints}"
EPOCHS="${4:-20}"
BATCH_SIZE="${5:-64}"
DEVICE="${6:-cpu}"
DDP_MODE="${7:-false}"
NPROC_PER_NODE="${8:-2}"
EXTRA_ARGS=("${@:9}")

if [ "$DDP_MODE" = "ddp" ] || [ "$DDP_MODE" = "true" ]; then
  torchrun --nproc_per_node="${NPROC_PER_NODE}" main.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --download-data \
    --save-dir "${SAVE_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --ddp \
    "${EXTRA_ARGS[@]}"
else
  python main.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --download-data \
    --save-dir "${SAVE_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}"
fi
