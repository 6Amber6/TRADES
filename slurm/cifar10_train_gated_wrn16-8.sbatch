#!/bin/bash
#SBATCH --job-name=c10-gated-16
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/cifar10_gated_wrn16-8_%j.out
#SBATCH --error=logs/cifar10_gated_wrn16-8_%j.err
#SBATCH --account=YOUR_ACCOUNT

# Modify --partition and --account for your cluster

cd ${SLURM_SUBMIT_DIR:-$HOME/TRADES}
mkdir -p logs

python train_cifar10/train_trades_cifar10_parallel_warmup_ema_aux_lrstage_gated.py \
  --epochs-sub 35 \
  --epochs-fusion 100 \
  --batch-size 128 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --epsilon 0.031 \
  --num-steps 10 \
  --step-size 0.007 \
  --beta 6.0 \
  --arch wrn16-8 \
  --model-dir ./model-cifar10-gated-wrn16-8
