"""PGD-20 robustness evaluation for CIFAR-100 4x25+unk routing model"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def unknown_routing_scores(unk_probs, a=1.0, b=0.5, T=1.0):
    B = unk_probs.size(0)
    n = unk_probs.size(1)
    conf = 1 - unk_probs
    other_unk = unk_probs.unsqueeze(2).expand(B, n, n)
    mask = 1 - torch.eye(n, device=unk_probs.device).unsqueeze(0).expand(B, n, n)
    other_unk = (other_unk * mask).sum(2) / (n - 1)
    scores = a * conf + b * other_unk
    w = F.softmax(scores / T, dim=1)
    return w


class UnknownRoutingFusion100(nn.Module):
    def __init__(self, submodels, owned_fine_per_group, a=1.0, b=0.5, T=1.0, margin=0.0):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        self.owned_fine = owned_fine_per_group
        self.a, self.b, self.T = a, b, T
        self.margin = margin

    def forward(self, x, return_aux=False):
        group_logits, unk_probs = [], []
        for m, owned in zip(self.submodels, self.owned_fine):
            logits = m(x)
            p = F.softmax(logits, dim=1)
            unk_probs.append(p[:, -1:])
            group_logits.append(logits)
        unk_probs = torch.cat(unk_probs, dim=1)
        w = unknown_routing_scores(unk_probs, a=self.a, b=self.b, T=self.T)
        B = group_logits[0].size(0)
        device = group_logits[0].device
        final = torch.zeros(B, 100, device=device, dtype=group_logits[0].dtype)
        for i, (loc, owned) in enumerate(zip(group_logits, self.owned_fine)):
            known = loc[:, :len(owned)]
            std = known.std(dim=1, keepdim=True).clamp(min=1e-6)
            known = known / std
            final[:, owned] += w[:, i:i+1] * known
        return (group_logits, final) if return_aux else final


def build_fine_classes_for_group(group_coarse):
    CIFAR100_FINE_TO_COARSE = [
        4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
        6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
        5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
        10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
        16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
    ]
    return [i for i, c in enumerate(CIFAR100_FINE_TO_COARSE) if c in group_coarse]


parser = argparse.ArgumentParser(description='PGD Attack CIFAR-100 4x25+unk routing')
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
parser.add_argument('--route-T', type=float, default=0.5)
parser.add_argument('--route-a', type=float, default=1.0)
parser.add_argument('--route-b', type=float, default=0.5)
parser.add_argument('--route-margin', type=float, default=0.0)
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
        group_order = ckpt.get("group_order", GROUP_ORDER)
        coarse_groups = ckpt.get("coarse_groups", COARSE_GROUPS)
        group_fine = ckpt.get("group_fine")
        saved_args = ckpt.get("args", {})
        route_a = saved_args.get("route_a", args.route_a)
        route_b = saved_args.get("route_b", args.route_b)
        route_T = saved_args.get("route_T_end", saved_args.get("route_T", args.route_T))
        route_margin = saved_args.get("route_margin", args.route_margin)
    else:
        raise ValueError("Checkpoint must contain state_dict")

    if group_fine is None:
        group_coarse = [coarse_groups[n] for n in group_order]
        group_fine = [build_fine_classes_for_group(g) for g in group_coarse]

    submodels = []
    for fine_classes in group_fine:
        m = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=len(fine_classes) + 1).to(device)
        submodels.append(m)

    model = UnknownRoutingFusion100(submodels, group_fine, a=route_a, b=route_b, T=route_T, margin=route_margin).to(device)
    model.load_state_dict(model_state)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    natural_err, robust_err = 0, 0
    total = 0
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
