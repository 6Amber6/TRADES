#!/bin/bash
#SBATCH --job-name=c100-gated3s-22
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd ${SLURM_SUBMIT_DIR:-$HOME/TRADES}
mkdir -p logs
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

stdbuf -oL python -u train_cifar100/train_trades_cifar100_gated_3stage_2group.py \
  --epochs-sub 100 \
  --epochs-router 15 \
  --epochs-fusion 100 \
  --batch-size 128 \
  --lr 0.1 \
  --router-lr 0.1 \
  --fusion-lr 0.1 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --epsilon 0.031 \
  --num-steps 10 \
  --step-size 0.007 \
  --beta 6.0 \
  --aux-weight 0.02 \
  --arch wrn22-8 \
  --model-dir ./model-cifar100-gated-3stage-2group-wrn22-8
