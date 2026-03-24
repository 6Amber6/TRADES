#!/bin/bash
#SBATCH --job-name=c100-aa-gated2g-22
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd ${SLURM_SUBMIT_DIR:-$HOME/TRADES}
mkdir -p logs
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

stdbuf -oL python -u auto_attack/eval_autoattack_cifar100_gated_2group.py \
  --model-path ./model-cifar100-gated-3stage-2group-wrn22-8/fusion-epoch90.pt \
  --data-dir ../data \
  --epsilon 0.03137 \
  --version standard \
  --batch-size 128 \
  --arch wrn22-8
