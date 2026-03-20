# CIFAR-10 Training SLURM Scripts

6 training jobs (3 methods × 2 architectures).

## Submit

```bash
cd /path/to/TRADES

# Submit all 6 (run from TRADES root)
sbatch slurm/cifar10_train_gated_wrn16-8.sbatch
sbatch slurm/cifar10_train_gated_wrn22-8.sbatch
sbatch slurm/cifar10_train_softrouting_wrn16-8.sbatch
sbatch slurm/cifar10_train_softrouting_wrn22-8.sbatch
sbatch slurm/cifar10_train_softrouting_conf_wrn16-8.sbatch
sbatch slurm/cifar10_train_softrouting_conf_wrn22-8.sbatch
```

## Customize

- `--partition`: Change to your cluster's GPU partition (e.g. `a100`, `v100`)
- `--account`: Replace `YOUR_ACCOUNT` with your SLURM account
- `--time`: WRN-16-8 ~8h, WRN-22-8 ~10h (adjust as needed)
- `cd`: Script uses `SLURM_SUBMIT_DIR` or `$HOME/TRADES`; ensure TRADES path is correct

## Logs

Output goes to `logs/cifar10_*_%j.out` and `logs/cifar10_*_%j.err` (%j = job ID).
