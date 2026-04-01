from __future__ import print_function
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


# =========================================================
# Meta-groups by coarse class IDs (must match training)
# =========================================================
COARSE_GROUPS = {
    "textured_organic": [7, 15, 16, 8, 13],
    "smooth_organic": [1, 0, 11, 12, 14],
    "rigid_manmade": [6, 5, 3, 18, 9],
    "large_structures": [10, 17, 19, 2, 4],
}
GROUP_ORDER = ["textured_organic", "smooth_organic", "rigid_manmade", "large_structures"]


# =========================================================
# Gated fusion model for 100 fine classes
# =========================================================
class GatedFusionWRN100(nn.Module):
    def __init__(self, submodels, num_classes=100, hidden_dim=256):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        emb_dim = self.submodels[0].nChannels
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * len(self.submodels), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(self.submodels))
        )
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x, return_aux=False):
        embs = []
        aux_outputs = []
        for m in self.submodels:
            emb, out = m(x, return_embedding=True)
            embs.append(emb)
            aux_outputs.append(out)
        gate_in = torch.cat(embs, dim=1)
        w = torch.softmax(self.gate(gate_in), dim=1)
        fused = torch.zeros_like(embs[0])
        for i, emb in enumerate(embs):
            fused = fused + w[:, i:i + 1] * emb
        out = self.fc(fused)
        if return_aux:
            return aux_outputs, out
        return out


# =========================================================
# Argument parser
# =========================================================
parser = argparse.ArgumentParser(description='PGD White-box Attack (CIFAR-100 Parallel Gated Fusion)')
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--no-cuda', action='store_true', default=False)

parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--step-size', type=float, default=0.003)

parser.add_argument('--random', action='store_true')
parser.add_argument('--no-random', dest='random', action='store_false')
parser.set_defaults(random=True)

parser.add_argument('--model-path', required=True)
parser.add_argument('--sub-depth', type=int, default=28, choices=[28, 34])
parser.add_argument('--sub-widen', type=int, default=8, choices=[4, 8, 10])

parser.add_argument('--ema', action='store_true',
                    help='evaluate using EMA weights (checkpoint should already contain EMA weights)')

args = parser.parse_args()

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


def build_fine_classes_for_group(group_coarse):
    return [i for i, c in enumerate(CIFAR100_FINE_TO_COARSE) if c in group_coarse]


# =========================================================
# Data loader
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
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
        group_names = checkpoint.get("group_order", GROUP_ORDER)
        coarse_groups = checkpoint.get("coarse_groups", COARSE_GROUPS)
    else:
        model_state = checkpoint
        group_names = GROUP_ORDER
        coarse_groups = COARSE_GROUPS

    group_coarse = [coarse_groups[name] for name in group_names]
    group_fine = [build_fine_classes_for_group(g) for g in group_coarse]

    submodels = []
    for fine_classes in group_fine:
        m = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=len(fine_classes)).to(device)
        submodels.append(m)

    model = GatedFusionWRN100(submodels, num_classes=100).to(device)
    model.load_state_dict(model_state)

    if args.ema:
        print('[INFO] Evaluating with EMA weights (assuming checkpoint contains EMA weights)')

    model.eval()
    print('PGD white-box attack (CIFAR-100 parallel gated fusion)')
    eval_adv_test_whitebox(model, test_loader)


if __name__ == '__main__':
    main()
