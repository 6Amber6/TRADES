#!/bin/bash
#SBATCH --job-name=c100-aa-4x25unk-22
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

stdbuf -oL python -u auto_attack/eval_autoattack_cifar100_4x25_unk.py \
  --model-path ./model-cifar100-4x25-unk-wrn22-8/fusion-epoch100.pt \
  --data-dir ./data \
  --epsilon 0.03137 \
  --version standard \
  --batch-size 128
