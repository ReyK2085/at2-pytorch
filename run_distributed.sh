#!/bin/bash
set -e
ARGS="$@"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

if [ "$GPU_COUNT" -eq 0 ]; then
  echo "❌ No GPUs detected. Exiting."
  exit 1
fi

echo "✅ Detected $GPU_COUNT GPUs"
echo "🚀 Launching distributed training with args: $ARGS"

LOG_DIR="logs"
mkdir -p $LOG_DIR
RUN_ID=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/run_${RUN_ID}.log"

torchrun --nproc_per_node=$GPU_COUNT train.py $ARGS | tee $LOG_FILE

echo "📄 Training complete. Logs saved to $LOG_FILE"
