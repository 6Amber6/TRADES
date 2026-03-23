"""
PGD White-box Attack evaluation for CIFAR-100 Gated Fusion (2 groups).
"""
from __future__ import print_function
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

from models.parallel_wrn import WRNWithEmbedding


# =========================================================
# CIFAR-100 Normalize
# =========================================================
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# =========================================================
# CIFAR-100 fine -> coarse mapping
# =========================================================
CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

COARSE_GROUPS = {
    "nature": [0, 1, 7, 8, 11, 12, 13, 15, 16, 2],
    "manmade": [3, 4, 5, 6, 9, 10, 14, 17, 18, 19],
}
GROUP_ORDER = ["nature", "manmade"]


def build_fine_classes_for_group(group_coarse):
    return sorted([i for i, c in enumerate(CIFAR100_FINE_TO_COARSE) if c in group_coarse])


# =========================================================
# Gated fusion model (must match training)
# =========================================================
class GatedFusionWRN100(nn.Module):
    def __init__(self, m_nature, m_manmade, emb_dim=640, hidden_dim=256):
        super().__init__()
        self.m_nature = m_nature
        self.m_manmade = m_manmade
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Linear(emb_dim, 100)

    def forward(self, x, return_aux=False):
        e_nat, out_nat = self.m_nature(x, return_embedding=True)
        e_man, out_man = self.m_manmade(x, return_embedding=True)
        gate_in = torch.cat([e_nat, e_man], dim=1)
        alpha = torch.sigmoid(self.gate(gate_in))
        emb = alpha * e_nat + (1 - alpha) * e_man
        out = self.fc(emb)
        if return_aux:
            return out_nat, out_man, out
        return out


# =========================================================
# Arguments
# =========================================================
parser = argparse.ArgumentParser(description='PGD White-box Attack (CIFAR-100 Gated 2-group)')
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--step-size', type=float, default=0.003)
parser.add_argument('--random', action='store_true', default=True)
parser.add_argument('--model-path', required=True)
parser.add_argument('--sub-depth', type=int, default=22)
parser.add_argument('--sub-widen', type=int, default=8)
parser.add_argument('--arch', default=None, choices=['wrn34-10', 'wrn22-8', 'wrn28-8'])
args = parser.parse_args()

ARCH_PRESETS = {'wrn34-10': (34, 10), 'wrn22-8': (22, 8), 'wrn28-8': (28, 8)}
if args.arch:
    args.sub_depth, args.sub_widen = ARCH_PRESETS[args.arch]

use_cuda = (not args.no_cuda) and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


# =========================================================
# Utils
# =========================================================
def _cifar_bounds(device_):
    mean = torch.tensor(CIFAR100_MEAN, device=device_).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=device_).view(1, 3, 1, 1)
    low = (0.0 - mean) / std
    high = (1.0 - mean) / std
    return low, high, std


def get_logits(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        return out[-1]
    return out


# =========================================================
# Data
# =========================================================
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

testset = torchvision.datasets.CIFAR100(
    root='../data', train=False, download=True, transform=transform_test
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
)


# =========================================================
# PGD White-box
# =========================================================
def pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random_start=True):
    model.eval()

    low, high, std = _cifar_bounds(X.device)
    eps = epsilon / std
    step = step_size / std

    logits = get_logits(model, X)
    err = (logits.argmax(1) != y).float().sum()

    X_pgd = Variable(X.data, requires_grad=True)
    if random_start:
        noise = torch.empty_like(X_pgd).uniform_(-1.0, 1.0) * eps
        X_pgd = Variable(X_pgd.data + noise, requires_grad=True)
        X_pgd = Variable(torch.max(torch.min(X_pgd, high), low), requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(get_logits(model, X_pgd), y)
        loss.backward()

        eta = step * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        delta = torch.clamp(X_pgd.data - X.data, -eps, eps)
        X_pgd = Variable(X.data + delta, requires_grad=True)
        X_pgd = Variable(torch.max(torch.min(X_pgd, high), low), requires_grad=True)

    err_pgd = (get_logits(model, X_pgd).argmax(1) != y).float().sum()
    print('err pgd (white-box): ', err_pgd)

    return err, err_pgd


def eval_adv_test_whitebox(model, loader):
    natural_err_total = 0
    robust_err_total = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        err_nat, err_adv = pgd_whitebox(
            model, x, y,
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            random_start=args.random
        )
        natural_err_total += err_nat
        robust_err_total += err_adv
        total += y.size(0)

    n = total
    print(f'natural_err_total: {natural_err_total}')
    print(f'robust_err_total: {robust_err_total}')
    print(f'natural_acc: {(n - natural_err_total) / n * 100:.2f}%')
    print(f'robust_acc: {(n - robust_err_total) / n * 100:.2f}%')


# =========================================================
# Main
# =========================================================
def main():
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
    else:
        model_state = checkpoint

    fine_nature = build_fine_classes_for_group(COARSE_GROUPS["nature"])
    fine_manmade = build_fine_classes_for_group(COARSE_GROUPS["manmade"])

    m_nature = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen,
                                 num_classes=len(fine_nature)).to(device)
    m_manmade = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen,
                                  num_classes=len(fine_manmade)).to(device)

    emb_dim = 64 * args.sub_widen
    model = GatedFusionWRN100(m_nature, m_manmade, emb_dim=emb_dim).to(device)
    model.load_state_dict(model_state)
    model.eval()

    print('PGD white-box attack (CIFAR-100 gated fusion, 2-group)')
    print(f'  Model: {args.model_path}')
    print(f'  Arch: WRN-{args.sub_depth}-{args.sub_widen} x2')
    print(f'  Nature classes: {len(fine_nature)}, Manmade classes: {len(fine_manmade)}')
    eval_adv_test_whitebox(model, test_loader)


if __name__ == '__main__':
    main()
