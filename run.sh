#!/usr/bin/env bash
set -e

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  cat <<'EOF'
Usage: ./run.sh [model] [dataset] [save_dir] [epochs] [batch_size] [device] [parallel] [nproc_per_node] [extra main.py args...]

Arguments:
  model           Model name (run `python main.py --mode list` for every registered model)
  dataset         Dataset name (cifar10, cifar100, stl10, imagenet, food101, flowers102, oxford_pets, imagefolder)
  save_dir        Directory to save checkpoints
  epochs          Number of training epochs
  batch_size      Batch size (total for single/dp, per process for ddp)
  device          Device string for PyTorch (cpu, cuda, cuda:0, ...)
  parallel        none (default) | dp (nn.DataParallel, one process) | ddp or true (torchrun, one process per GPU)
  nproc_per_node  Number of processes / GPUs for ddp

Anything after the 8th argument is forwarded to main.py.

Example:
  ./run.sh mobilevit cifar10 ./checkpoints 20 64 cpu
  ./run.sh resnet50 imagenet ./checkpoints 90 256 cuda dp 1 --data-root /data/imagenet --amp
  ./run.sh resnet50 imagenet ./checkpoints 90 64 cuda ddp 4 --data-root /data/imagenet --amp
  ./run.sh vit_tiny cifar10 ./checkpoints 100 128 cuda none 1 --optimizer adamw --lr 1e-3 --warmup-epochs 5 --amp
EOF
  exit 0
fi

MODEL="${1:-mobilevit}"
DATASET="${2:-cifar10}"
SAVE_DIR="${3:-./checkpoints}"
EPOCHS="${4:-20}"
BATCH_SIZE="${5:-64}"
DEVICE="${6:-cpu}"
PARALLEL="${7:-none}"
NPROC_PER_NODE="${8:-2}"
EXTRA_ARGS=("${@:9}")

COMMON_ARGS=(
  --model "${MODEL}"
  --dataset "${DATASET}"
  --download-data
  --save-dir "${SAVE_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
)

case "${PARALLEL}" in
  ddp|true)
    torchrun --nproc_per_node="${NPROC_PER_NODE}" main.py "${COMMON_ARGS[@]}" --parallel ddp "${EXTRA_ARGS[@]}"
    ;;
  dp)
    python main.py "${COMMON_ARGS[@]}" --device "${DEVICE}" --parallel dp "${EXTRA_ARGS[@]}"
    ;;
  *)
    python main.py "${COMMON_ARGS[@]}" --device "${DEVICE}" "${EXTRA_ARGS[@]}"
    ;;
esac
