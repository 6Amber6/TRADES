#!/bin/bash
#SBATCH --job-name=c100-parallel-34
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd ${SLURM_SUBMIT_DIR:-$HOME/TRADES}
mkdir -p logs
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

stdbuf -oL python -u train_cifar100/train_trades_cifar100_parallel_warmup_ema_aux_lrstage.py \
  --epochs-sub 80 \
  --epochs-fusion 100 \
  --batch-size 128 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --epsilon 0.031 \
  --num-steps 10 \
  --step-size 0.007 \
  --beta 6.0 \
  --sub-depth 34 \
  --sub-widen 10 \
  --scheduler cosine \
  --model-dir ./model-cifar100-parallel-wrn34-10 \
  --resume
