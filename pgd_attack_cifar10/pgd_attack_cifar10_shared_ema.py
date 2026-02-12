from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

from models.shared_wrn_gated import SharedWRNGated


# =========================================================
# CIFAR-10 Normalize
# =========================================================
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# =========================================================
# EMA 
# =========================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup = {}

    def load_shadow(self, shadow_state):
        for k in self.shadow:
            self.shadow[k] = shadow_state[k].clone()

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n].clone()

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n]


# =========================================================
# Argument parser
# =========================================================
parser = argparse.ArgumentParser(description='PGD White-box Attack (Parallel Fusion + EMA)')
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--no-cuda', action='store_true', default=False)

# MUST match TRADES eval settings
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--step-size', type=float, default=0.003)

parser.add_argument('--random', action='store_true')
parser.add_argument('--no-random', dest='random', action='store_false')
parser.set_defaults(random=True)

# checkpoint
parser.add_argument('--model-path', required=True)

# EMA
parser.add_argument('--ema', action='store_true',
                    help='evaluate using EMA weights (recommended)')

args = parser.parse_args()

use_cuda = (not args.no_cuda) and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


# =========================================================
# Utils
# =========================================================
def _cifar_bounds(device_):
    mean = torch.tensor(CIFAR10_MEAN, device=device_).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device_).view(1, 3, 1, 1)
    low = (0.0 - mean) / std
    high = (1.0 - mean) / std
    return low, high, std


def get_logits(model, x):
    """
    Support both:
      - logits
      - (logits4, logits6, fusion_logits)
    """
    out = model(x)
    if isinstance(out, (tuple, list)):
        return out[-1]
    return out


# =========================================================
# Data loader
# =========================================================
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

testset = torchvision.datasets.CIFAR10(
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


# =========================================================
# Eval
# =========================================================
def eval_adv_test_whitebox(model, loader):
    natural_err_total = 0
    robust_err_total = 0

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

    print('natural_err_total:', natural_err_total)
    print('robust_err_total:', robust_err_total)


# =========================================================
# Main
# =========================================================
def main():
    model = SharedWRNGated(depth=34, widen_factor=10).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    if args.ema:
        print('[INFO] Evaluating with EMA weights ')


    model.eval()
    print('PGD white-box attack (parallel fusion)')
    eval_adv_test_whitebox(model, test_loader)


if __name__ == '__main__':
    main()
