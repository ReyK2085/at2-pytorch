# CIFAR-10 Distributed Training

This repository contains two main files:

- **train.py**: A PyTorch training script for CIFAR-10.
- **run_distributed.sh**: A shell script to launch distributed training with `torchrun`.

---

## 🚀 Usage

### Single-GPU Training
Run the training script directly:
```bash
python3 train.py --epochs 10 --batch-size 64 --lr 0.001 --tracker wandb
