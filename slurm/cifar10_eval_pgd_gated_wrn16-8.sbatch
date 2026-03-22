#!/bin/bash
#SBATCH --job-name=c10-pgd-gated-16
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd ${SLURM_SUBMIT_DIR:-$HOME/TRADES}
mkdir -p logs
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

stdbuf -oL python -u pgd_attack_cifar10/pgd_attack_cifar10_parallel_gated_ema.py \
  --model-path ./model-cifar10-gated-wrn16-8/fusion-epoch95.pt \
  --wrn4-path ./model-cifar10-gated-wrn16-8/wrn4_final.pt \
  --wrn6-path ./model-cifar10-gated-wrn16-8/wrn6_final.pt \
  --epsilon 0.031 \
  --num-steps 20 \
  --step-size 0.003 \
  --arch wrn16-8
