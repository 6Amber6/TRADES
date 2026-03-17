"""
AutoAttack evaluation for CIFAR-100 4x25+unk routing (unk_routing_v2.py).
Model expects normalized input. Wraps with NormalizedModel for [0,1] space.
"""

from __future__ import print_function
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    import torch.nn.functional as F
    B = unk_probs.size(0)
    n = unk_probs.size(1)
    conf = 1 - unk_probs
    other_unk = unk_probs.unsqueeze(2).expand(B, n, n)
    mask = 1 - torch.eye(n, device=unk_probs.device).unsqueeze(0).expand(B, n, n)
    other_unk = (other_unk * mask).sum(2) / (n - 1)
    scores = a * conf + b * other_unk
    return F.softmax(scores / T, dim=1)


class UnknownRoutingFusion100(nn.Module):
    def __init__(self, submodels, owned_fine_per_group, a=1.0, b=0.5, T=1.0, margin=0.0):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        self.owned_fine = owned_fine_per_group
        self.a, self.b, self.T = a, b, T
        self.margin = margin

    def forward(self, x, return_aux=False):
        import torch.nn.functional as F
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


class NormalizedModel(nn.Module):
    """Wraps model so AutoAttack operates in [0,1], model receives normalized input."""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def main():
    parser = argparse.ArgumentParser(description='AutoAttack eval – CIFAR-100 4x25+unk routing')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-dir', default='../data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epsilon', type=float, default=8/255)
    parser.add_argument('--version', default='standard', choices=['standard', 'plus', 'rand'])
    parser.add_argument('--sub-depth', type=int, default=34, help='WRN depth (must match training)')
    parser.add_argument('--sub-widen', type=int, default=10, help='WRN widen factor (must match training)')
    parser.add_argument('--route-T', type=float, default=0.5)
    parser.add_argument('--route-a', type=float, default=1.0)
    parser.add_argument('--route-b', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--log-path', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu')
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

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
        sub_depth = saved_args.get("sub_depth", args.sub_depth)
        sub_widen = saved_args.get("sub_widen", args.sub_widen)
    else:
        raise ValueError("Checkpoint must contain state_dict")

    if group_fine is None:
        group_coarse = [coarse_groups[n] for n in group_order]
        group_fine = [build_fine_classes_for_group(g) for g in group_coarse]

    submodels = []
    for fine_classes in group_fine:
        m = WRNWithEmbedding(depth=sub_depth, widen_factor=sub_widen,
                             num_classes=len(fine_classes) + 1).to(device)
        submodels.append(m)

    fusion = UnknownRoutingFusion100(submodels, group_fine, a=route_a, b=route_b, T=route_T).to(device)
    fusion.load_state_dict(model_state)
    model = NormalizedModel(fusion, CIFAR100_MEAN, CIFAR100_STD).to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    x_test = torch.cat([x for x, _ in loader], dim=0).to(device)
    y_test = torch.cat([y for _, y in loader], dim=0).to(device)

    if args.log_path is None:
        model_dir = os.path.dirname(args.model_path)
        args.log_path = os.path.join(model_dir, 'log-aa-4x25-unk-v2.log')

    try:
        from autoattack import AutoAttack
    except ImportError:
        raise ImportError('pip install autoattack')

    adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, log_path=args.log_path,
                           version=args.version, seed=args.seed)

    print(f'Model: {args.model_path}')
    print(f'Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.2f}/255)')
    print(f'Version: {args.version}')
    print(f'Samples: {len(x_test)}')

    t0 = time.time()
    adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
    print(f'Completed in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
