# TRADES - Adversarial Robustness via TRADES Loss

## Project Overview

TRADES (TRadeoff-inspired Adversarial Defense via Surrogate-loss minimization) 是一个基于 PyTorch 的对抗训练框架，核心思想是通过 KL 散度正则化平衡自然准确率与对抗鲁棒性。源自 ICML'19 论文 (Hongyang Zhang et al.)。

核心损失函数: `L(f(X), Y) + β * max KL(f(X), f(X'))`

## Project Structure

```
trades.py                    # 核心 TRADES loss 实现
models/                      # 模型定义
  small_cnn.py               #   MNIST 用 SmallCNN
  net_mnist.py               #   MNIST 用 Net / Net_binary
  wideresnet.py              #   WideResNet-34-10 (CIFAR 主力模型)
  wideresnet_update.py       #   WideResNet 更新版
  wideresnet_update_gn.py    #   WideResNet + GroupNorm
  resnet.py                  #   标准 ResNet
  parallel_wrn.py            #   ParallelFusionWRN (双路拼接融合)
  shared_wrn_gated.py        #   共享骨干 + 门控融合
  parallel_wrn_gn.py         #   并行 WRN + GroupNorm
train_cifar10/               # CIFAR-10 训练脚本 (多种变体)
train_cifar100/              # CIFAR-100 训练脚本
pgd_attack_cifar10/          # PGD 攻击评估 (CIFAR-10)
pgd_attack_cifar100/         # PGD 攻击评估 (CIFAR-100)
auto_attack/                 # AutoAttack 标准评估
```

## Dependencies

- Python 3.6+
- PyTorch 1.x+ with CUDA
- torchvision
- numpy
- autoattack (`pip install autoattack`，用于 AutoAttack 评估)

## Key Hyperparameters

| 参数 | MNIST | CIFAR-10/100 |
|------|-------|-------------|
| epsilon | 0.3 | 0.031 |
| step_size | 0.01 | 0.007 |
| num_steps | 40 | 10 |
| beta | 1.0 | 6.0 |
| lr | 0.01 | 0.1 |
| batch_size | 128 | 128 |
| weight_decay | - | 2e-4 ~ 5e-4 |

## Common Commands

### Training

从子目录运行训练脚本时需设置 PYTHONPATH：

```bash
# MNIST
python train_trades_mnist.py --epochs 100 --beta 1.0

# CIFAR-10 基础版
cd train_cifar10
PYTHONPATH=.. python train_trades_cifar10.py \
  --epochs 76 --batch-size 128 --lr 0.1 \
  --epsilon 0.031 --num-steps 10 --step-size 0.007 \
  --beta 6.0 --model-dir ./model-cifar-wideResNet

# CIFAR-10 并行融合 (两阶段)
cd train_cifar10
PYTHONPATH=.. python train_trades_cifar10_parallel_warmup_ema_aux_lrstage.py \
  --epochs-sub 100 --epochs-fusion 100 ...
```

### Evaluation

```bash
# PGD 攻击评估
cd pgd_attack_cifar10
PYTHONPATH=.. python pgd_attack_cifar10.py --model-path <checkpoint>

# AutoAttack 评估
python auto_attack/eval_autoattack_cifar10.py \
  --model-path <checkpoint> --data-dir ../data \
  --epsilon 0.03137 --version standard --batch-size 128
```

## Architecture Variants

项目探索了多种模型融合策略（主要针对 CIFAR-10/100）：

1. **基础单模型**: WideResNet-34-10 + TRADES loss
2. **并行融合 (Parallel Fusion)**: 两个 WRN 分别处理不同类别子集，拼接特征后融合
3. **门控融合 (Gated Fusion)**: 学习 sigmoid 门控权重进行融合
4. **软路由 (Soft Routing)**: 基于 unknown score 的非对称加权
5. **置信路由 (Confidence Routing)**: 基于置信度的简化路由

## Checkpoint Formats

项目中存在两种 checkpoint 格式：

```python
# 简单格式 - 仅 state_dict
torch.save(model.state_dict(), path)

# 完整格式 - 包含优化器、调度器、EMA 等
checkpoint = {
    'model_state_dict': ...,
    'm4_state_dict': ...,
    'm6_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'ema_shadow': ...,
    'stage2_epoch': ...
}
```

## Coding Conventions

- 数据目录默认 `../data`，checkpoint 通过 `--model-dir` 指定
- 训练脚本遵循统一模式: `train()` → `eval_train()` → `eval_test()` → `adjust_learning_rate()`
- 数据输入为 raw [0,1] 像素值（无 mean/std 归一化，trades.py 中可选配置）
- 所有脚本通过 argparse 接收命令行参数
- 从 epoch 60 开始每轮保存 checkpoint
