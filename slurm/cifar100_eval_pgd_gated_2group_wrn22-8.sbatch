#!/bin/bash
#SBATCH --job-name=c100-pgd-gated2g-22
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

stdbuf -oL python -u pgd_attack_cifar100/pgd_attack_cifar100_gated_2group.py \
  --model-path ./model-cifar100-gated-3stage-2group-wrn22-8/fusion-epoch70.pt \
  --epsilon 0.031 \
  --num-steps 20 \
  --step-size 0.003 \
  --arch wrn22-8
