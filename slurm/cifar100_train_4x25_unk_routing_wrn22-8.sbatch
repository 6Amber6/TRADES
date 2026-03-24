#!/bin/bash
#SBATCH --job-name=c100-4x25unk-22
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd ${SLURM_SUBMIT_DIR:-$HOME/TRADES}
mkdir -p logs
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

stdbuf -oL python -u train_cifar100/train_cifar100_4x25_unk_routing.py \
  --epochs-sub 60 \
  --epochs-fusion 100 \
  --batch-size 128 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --epsilon 0.031 \
  --num-steps 10 \
  --step-size 0.007 \
  --beta 6.0 \
  --sub-depth 22 \
  --sub-widen 8 \
  --aux-weight 0.05 \
  --aux-unknown-weight 0.3 \
  --route-a 1.0 \
  --route-b 0.5 \
  --route-T-start 1.0 \
  --route-T-end 0.2 \
  --scheduler step \
  --model-dir ./model-cifar100-4x25-unk-wrn22-8
