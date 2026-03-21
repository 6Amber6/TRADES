#!/bin/bash
#SBATCH --job-name=c10-gated-16
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd ${SLURM_SUBMIT_DIR:-$HOME/TRADES}
mkdir -p logs

python train_cifar10/train_trades_cifar10_parallel_warmup_ema_aux_lrstage_gated.py \
  --epochs-sub 35 \
  --epochs-fusion 100 \
  --batch-size 128 \
  --lr 0.1 \
  --fusion-lr 0.03 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --epsilon 0.031 \
  --num-steps 10 \
  --step-size 0.007 \
  --beta 6.0 \
  --arch wrn16-8 \
  --model-dir ./model-cifar10-gated-wrn16-8 \
  --resume ""
