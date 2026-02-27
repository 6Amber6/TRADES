"""PGD evaluation for CIFAR-100 Parallel Gated v2 (proj + dropout + BN)"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import transforms

from models.parallel_wrn import WRNWithEmbedding


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

COARSE_GROUPS = {
    "textured_organic": [7, 15, 16, 8, 13],
    "smooth_organic": [1, 0, 11, 12, 14],
    "rigid_manmade": [6, 5, 3, 18, 9],
    "large_structures": [10, 17, 19, 2, 4],
}
GROUP_ORDER = ["textured_organic", "smooth_organic", "rigid_manmade", "large_structures"]

CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


class GatedFusionWRN100(nn.Module):
    def __init__(self, submodels, num_classes=100, hidden_dim=256, proj_dim=64):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        emb_dim = self.submodels[0].nChannels
        n_experts = len(self.submodels)
        self.proj = nn.ModuleList([nn.Linear(emb_dim, proj_dim) for _ in range(n_experts)])
        gate_in_dim = proj_dim * n_experts
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_experts)
        )
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x, return_aux=False):
        embs = []
        aux_outputs = []
        for m in self.submodels:
            emb, out = m(x, return_embedding=True)
            embs.append(emb)
            aux_outputs.append(out)
        proj_embs = [self.proj[i](embs[i]) for i in range(len(embs))]
        gate_in = torch.cat(proj_embs, dim=1)
        w = torch.softmax(self.gate(gate_in), dim=1)
        fused = torch.zeros_like(embs[0])
        for i, emb in enumerate(embs):
            fused = fused + w[:, i:i + 1] * emb
        out = self.fc(fused)
        if return_aux:
            return aux_outputs, out
        return out


def build_fine_classes_for_group(group_coarse):
    return [i for i, c in enumerate(CIFAR100_FINE_TO_COARSE) if c in group_coarse]


parser = argparse.ArgumentParser(description='PGD CIFAR-100 Parallel Gated v2')
parser.add_argument('--model-path', required=True)
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--step-size', type=float, default=0.003)
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--random', action='store_true', default=True)
parser.add_argument('--no-random', dest='random', action='store_false')
parser.add_argument('--sub-depth', type=int, default=28)
parser.add_argument('--sub-widen', type=int, default=8)
args = parser.parse_args()

device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}


def _cifar_bounds(dev):
    mean = torch.tensor(CIFAR100_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=dev).view(1, 3, 1, 1)
    return (0.0 - mean) / std, (1.0 - mean) / std, std


def get_logits(model, x):
    out = model(x)
    return out[-1] if isinstance(out, (tuple, list)) else out


def pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random_start=True):
    model.eval()
    low, high, std = _cifar_bounds(X.device)
    eps = epsilon / std
    step = step_size / std
    err_nat = (get_logits(model, X).argmax(1) != y).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random_start:
        noise = torch.empty_like(X_pgd).uniform_(-1.0, 1.0) * eps
        X_pgd = Variable(torch.max(torch.min(X_pgd.data + noise, high), low), requires_grad=True)
    for _ in range(num_steps):
        if X_pgd.grad is not None:
            X_pgd.grad.zero_()
        loss = nn.CrossEntropyLoss()(get_logits(model, X_pgd), y)
        loss.backward()
        eta = step * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        delta = torch.clamp(X_pgd.data - X.data, -eps, eps)
        X_pgd = Variable(torch.max(torch.min(X.data + delta, high), low), requires_grad=True)
    err_pgd = (get_logits(model, X_pgd).argmax(1) != y).float().sum()
    return err_nat, err_pgd


def main():
    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model_state = ckpt["state_dict"]
        group_names = ckpt.get("group_order", GROUP_ORDER)
        coarse_groups = ckpt.get("coarse_groups", COARSE_GROUPS)
    else:
        model_state = ckpt
        group_names = GROUP_ORDER
        coarse_groups = COARSE_GROUPS
    group_coarse = [coarse_groups[n] for n in group_names]
    group_fine = [build_fine_classes_for_group(g) for g in group_coarse]
    submodels = []
    for fine_classes in group_fine:
        m = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=len(fine_classes)).to(device)
        submodels.append(m)
    model = GatedFusionWRN100(submodels, num_classes=100).to(device)
    model.load_state_dict(model_state)
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    natural_err, robust_err, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        en, er = pgd_whitebox(model, x, y, args.epsilon, args.num_steps, args.step_size, args.random)
        natural_err += en.item()
        robust_err += er.item()
        total += x.size(0)
    print(f'Natural acc: {(total - natural_err) / total * 100:.2f}%')
    print(f'PGD-{args.num_steps} robust acc: {(total - robust_err) / total * 100:.2f}%')


if __name__ == '__main__':
    main()
