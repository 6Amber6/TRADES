# Parallel Expert Fusion for Adversarial Robustness

This project builds upon [TRADES](https://arxiv.org/pdf/1901.08573.pdf) (ICML 2019) to investigate whether **parallel sub-network specialization with fusion** can improve the robustness–accuracy trade-off in adversarial training.

## Motivation

Standard adversarial training (e.g., TRADES) uses a single monolithic network to learn robust representations across all classes simultaneously. We hypothesize that **splitting the classification task among specialized expert sub-networks**, each responsible for a semantically coherent subset of classes, and then **fusing their outputs** can yield:

1. **Higher clean accuracy** — each expert focuses on a smaller, more coherent label space.
2. **Improved adversarial robustness** — specialized experts produce more discriminative features that are harder to fool.
3. **Flexible routing** — learning to weight expert opinions based on per-sample confidence can further improve performance under adversarial perturbation.

We explore four fusion strategies on CIFAR-10 and two on CIFAR-100, all built on top of WideResNet-34-10 sub-networks trained with the TRADES loss.

## Method Overview

### Architecture

Each expert is a `WRNWithEmbedding` (WideResNet-34-10), producing both logits and an embedding vector (dim = 640). Training proceeds in two stages:

- **Stage 1 (Sub-network Training):** Each expert is adversarially trained (TRADES) independently on its own class subset with an additional *unknown* class for out-of-distribution inputs. EMA is applied for stabilization.
- **Stage 2 (Fusion Training):** Experts are combined via one of the fusion strategies below, and the fusion module is adversarially fine-tuned end-to-end with a reduced backbone learning rate.

### CIFAR-10: Two Experts

Classes are split into **Vehicles** {airplane, automobile, ship, truck} and **Animals** {bird, cat, deer, dog, frog, horse}. Two experts (M4 and M6) are trained, one per group.

| Method | Fusion Mechanism |
|--------|-----------------|
| **Concat + FC** | `emb = [e₄ ; e₆]`, `logits = FC(emb)` |
| **Feature-level Soft Routing (Gated)** | `α = sigmoid(MLP([e₄ ; e₆]))`, `emb = α · e₄ + (1−α) · e₆`, `logits = FC(emb)` |
| **a,b Routing** | `score₄ = a · (1 − unk₄) + b · unk₆`, `score₆ = a · (1 − unk₆) + b · unk₄`, `[w₄, w₆] = softmax([score₄, score₆])`, `logits = w₄ · logits₄ + w₆ · logits₆` |
| **Confidence-only Routing** | `score₄ = 1 − unk₄`, `score₆ = 1 − unk₆`, `[w₄, w₆] = softmax([score₄, score₆])`, `logits = w₄ · logits₄ + w₆ · logits₆` |

### CIFAR-100: Four Experts

100 fine classes are grouped into 4 semantically coherent super-groups of 25 classes each (textured organic, smooth organic, rigid man-made, large structures). Each expert handles 25 classes + 1 unknown class. Routing follows the same a,b / confidence-only paradigm.

## Results

### CIFAR-10 (WideResNet-34-10, ε = 8/255, β = 6.0)

| Method | Params | Clean Acc (%) | PGD-20 Acc (%) | AutoAttack Acc (%) |
|--------|--------|:---:|:---:|:---:|
| TRADES Baseline | 46.2M | 84.62 | 55.30 | 51.31 |
| Concat + FC (M4 M6) | 92.4M+ | 88.75 / 88.43 | 59.32 / 59.14 | 53.36 (ep100) |
| Feature-level Gated (M4 M6) | 92.4M+ | 88.95 / 89.13 | 58.43 / 58.71 | 53.02 (ep90) |
| a,b Routing (M5 M7) | 92.4M+ | 87.75 / 87.71 | 58.88 / 58.78 | 53.18 (ep95) |
| Confidence-only Routing (M5 M7) | 92.4M+ | 87.78 / 88.04 | 58.26 / 58.74 | **53.66** (ep85) |

> Two values per cell correspond to different evaluation epochs / runs. AutoAttack results report the best epoch.

### CIFAR-100 (WideResNet-34-10, ε = 8/255, β = 6.0)

| Method | Clean Acc (%) | PGD-20 Acc (%) | AutoAttack Acc (%) |
|--------|:---:|:---:|:---:|
| TRADES Baseline | 60.37 | 32.42 | 27.08 |

## Key Findings

- All parallel expert methods outperform the single-model TRADES baseline in clean accuracy (+3~4.5%), PGD-20 robustness (+3~4%), and AutoAttack accuracy (+1.7~2.3%).
- **Confidence-only routing** achieves the best AutoAttack accuracy (53.66%) with the simplest routing mechanism, indicating that complex routing may not be necessary.

## Project Structure

```
trades.py                           # Core TRADES loss implementation
models/
  wideresnet.py                     #   WideResNet (original TRADES)
  wideresnet_update.py              #   WideResNet (updated, used by parallel models)
  parallel_wrn.py                   #   WRNWithEmbedding + ParallelFusionWRN
  resnet.py                         #   Standard ResNet

train_cifar10/
  train_cifar10_baseline.py         # TRADES baseline
  train_cifar10_concat.py           # Concat + FC
  train_cifar10_gated.py            # Feature-level gated
  train_cifar10_routing_ab.py       # a,b routing
  train_cifar10_routing_conf.py     # Confidence-only routing

train_cifar100/
  train_cifar100_baseline.py        # TRADES baseline
  train_cifar100_concat.py          # Concat + FC
  train_cifar100_gated.py           # Feature-level gated
  train_cifar100_routing_ab.py      # a,b routing (4 experts)
  train_cifar100_routing_uniform.py # Uniform routing (4 experts)

pgd_attack_cifar10/                 # PGD-20 evaluation (eval_pgd_cifar10_*.py)
pgd_attack_cifar100/                # PGD-20 evaluation (eval_pgd_cifar100_*.py)
auto_attack/                        # AutoAttack evaluation (eval_aa_*.py)
slurm/                              # SLURM job scripts (WRN-34-10)
```

## Requirements

- Python 3.8+
- PyTorch 1.x+ with CUDA
- torchvision
- numpy
- [autoattack](https://github.com/fra31/auto-attack) (`pip install autoattack`)

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Architecture | WideResNet-34-10 per expert |
| ε (L∞) | 8/255 ≈ 0.031 |
| PGD steps (train) | 10 |
| Step size | 0.007 |
| β (TRADES) | 6.0 |
| Batch size | 128 |
| Learning rate | 0.1 (cosine / multi-step decay) |
| Weight decay | 5×10⁻⁴ |
| EMA decay | 0.999 |

## Reference

This project builds on the official TRADES implementation:

```bibtex
@inproceedings{zhang2019theoretically,
    author    = {Hongyang Zhang and Yaodong Yu and Jiantao Jiao and
                 Eric P. Xing and Laurent El Ghaoui and Michael I. Jordan},
    title     = {Theoretically Principled Trade-off between Robustness and Accuracy},
    booktitle = {International Conference on Machine Learning (ICML)},
    year      = {2019}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
