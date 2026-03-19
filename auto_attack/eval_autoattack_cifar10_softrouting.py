"""
AutoAttack evaluation for SoftRoutingFusion (train_trades_cifar10_softrouting.py).
This version uses the (a, b) asymmetric routing weights.

Run from repo root:
  python auto_attack/eval_autoattack_cifar10_softrouting.py \
      --model-path ./train_cifar10/model-parallel/fusion-epoch100.pt

Checkpoint formats accepted:
  - fusion-epoch{N}.pt  : raw state_dict
  - checkpoint-last.pt  : dict with 'model_state_dict' key

Routing hyperparameters (--route-a, --route-b, --route-T, --route-margin) must match
the values used during training.
"""

from __future__ import print_function
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.parallel_wrn import WRNWithEmbedding

# ---- constants ----
CIFAR10_MEAN    = (0.4914, 0.4822, 0.4465)
CIFAR10_STD     = (0.2023, 0.1994, 0.2010)
VEHICLE_CLASSES = [0, 1, 8, 9]
ANIMAL_CLASSES  = [2, 3, 4, 5, 6, 7]


# ---- routing helpers (must match training script exactly) ----
def map_expert_to_10(logits_local, owned_classes, num_classes=10, margin=0.0):
    B = logits_local.size(0)
    K = len(owned_classes)
    known = logits_local[:, :K]
    unk   = logits_local[:, K:K+1]
    logits_10 = (unk - margin).expand(B, num_classes).clone()
    for i, c in enumerate(owned_classes):
        logits_10[:, c] = known[:, i]
    return logits_10


def soft_routing_fusion(logits4_local, logits6_local, a=1.0, b=0.5, T=1.0, margin=0.0):
    p4 = F.softmax(logits4_local, dim=1)
    p6 = F.softmax(logits6_local, dim=1)
    unk4 = p4[:, -1:]
    unk6 = p6[:, -1:]
    conf4 = 1 - unk4
    conf6 = 1 - unk6
    s4 = a * conf4 + b * unk6
    s6 = a * conf6 + b * unk4
    s  = torch.cat([s4, s6], dim=1)
    w  = F.softmax(s / T, dim=1)
    w4, w6 = w[:, :1], w[:, 1:]
    logits4_10 = map_expert_to_10(logits4_local, VEHICLE_CLASSES, margin=margin)
    logits6_10 = map_expert_to_10(logits6_local, ANIMAL_CLASSES,  margin=margin)
    return w4 * logits4_10 + w6 * logits6_10, w4, w6


class SoftRoutingFusion(nn.Module):
    def __init__(self, m4, m6, a=1.0, b=0.5, T=1.0, margin=0.0):
        super().__init__()
        self.m4 = m4
        self.m6 = m6
        self.a  = a
        self.b  = b
        self.T  = T
        self.margin = margin

    def forward(self, x, return_aux=False):
        out4 = self.m4(x)
        out6 = self.m6(x)
        logits4 = out4[0] if isinstance(out4, (tuple, list)) else out4
        logits6 = out6[0] if isinstance(out6, (tuple, list)) else out6
        final_logits, _, _ = soft_routing_fusion(
            logits4, logits6, a=self.a, b=self.b, T=self.T, margin=self.margin)
        if return_aux:
            return logits4, logits6, final_logits
        return final_logits


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def main():
    parser = argparse.ArgumentParser(
        description='AutoAttack eval – SoftRoutingFusion (a/b variant)')
    parser.add_argument('--model-path',    required=True)
    parser.add_argument('--data-dir',      default='../data')
    parser.add_argument('--batch-size',    type=int,   default=128)
    parser.add_argument('--epsilon',       type=float, default=8/255)
    parser.add_argument('--version',       default='standard',
                        choices=['standard', 'plus', 'rand'])
    parser.add_argument('--sub-depth',     type=int,   default=34,
                        help='WRN depth (34=WRN-34-10, 16=WRN-16-8)')
    parser.add_argument('--sub-widen',     type=int,   default=10,
                        help='WRN widen factor (10=WRN-34-10, 8=WRN-16-8)')
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--no-cuda',       action='store_true')
    parser.add_argument('--log-path',      default=None)
    # routing hyperparameters – must match training
    parser.add_argument('--route-a',       type=float, default=1.0)
    parser.add_argument('--route-b',       type=float, default=0.5)
    parser.add_argument('--route-T',       type=float, default=1.0)
    parser.add_argument('--route-margin',  type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(
        'cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu')
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # ---- data ----
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    x_test = torch.cat([x for x, _ in test_loader], dim=0).to(device)
    y_test = torch.cat([y for _, y in test_loader], dim=0).to(device)

    # ---- model ----
    # num_classes=5 (4 vehicle + unknown) and 7 (6 animal + unknown)
    m4 = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=5).to(device)
    m6 = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=7).to(device)
    fusion = SoftRoutingFusion(
        m4, m6,
        a=args.route_a, b=args.route_b,
        T=args.route_T, margin=args.route_margin
    ).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        fusion.load_state_dict(ckpt['model_state_dict'])
    else:
        fusion.load_state_dict(ckpt)

    model = NormalizedModel(fusion, CIFAR10_MEAN, CIFAR10_STD).to(device)
    model.eval()

    # ---- log path ----
    if args.log_path is None:
        model_dir = os.path.dirname(args.model_path)
        args.log_path = os.path.join(model_dir, 'log-aa-softrouting.log')

    # ---- AutoAttack ----
    try:
        from autoattack import AutoAttack
    except ImportError:
        raise ImportError('pip install autoattack')

    adversary = AutoAttack(
        model, norm='Linf', eps=args.epsilon,
        log_path=args.log_path, version=args.version, seed=args.seed)

    print(f'Model  : {args.model_path}')
    print(f'Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.2f}/255)')
    print(f'Version: {args.version}')
    print(f'Routing: a={args.route_a}, b={args.route_b}, T={args.route_T}, margin={args.route_margin}')
    print(f'Samples: {len(x_test)}')

    t0 = time.time()
    adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
    print(f'Completed in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
