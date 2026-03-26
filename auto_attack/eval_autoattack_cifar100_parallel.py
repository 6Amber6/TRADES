"""
AutoAttack evaluation for CIFAR-100 ParallelFusionWRN100 model.
Model expects normalized input. Wraps with NormalizedModel so AutoAttack
operates in [0,1] space.

Usage:
    python auto_attack/eval_autoattack_cifar100_parallel.py \
        --model-path train_cifar100/model-cifar100-parallel/fusion-ema-epoch100.pt \
        --sub-depth 28 --sub-widen 8
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

CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

COARSE_GROUPS = {
    "textured_organic": [7, 15, 16, 8, 13],
    "smooth_organic": [1, 0, 11, 12, 14],
    "rigid_manmade": [6, 5, 3, 18, 9],
    "large_structures": [10, 17, 19, 2, 4],
}
GROUP_ORDER = ["textured_organic", "smooth_organic", "rigid_manmade", "large_structures"]


class ParallelFusionWRN100(nn.Module):
    def __init__(self, submodels, num_classes=100):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        emb_dim = sum(m.nChannels for m in self.submodels)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        embs = []
        for m in self.submodels:
            emb, _ = m(x, return_embedding=True)
            embs.append(emb)
        return self.fc(torch.cat(embs, dim=1))


class NormalizedModel(nn.Module):
    """Wraps model so AutoAttack operates in [0,1], model receives normalized input."""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def build_fine_classes_for_group(group_coarse):
    return [i for i, c in enumerate(CIFAR100_FINE_TO_COARSE) if c in group_coarse]


def main():
    parser = argparse.ArgumentParser(description='AutoAttack eval - CIFAR-100 ParallelFusionWRN100')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-dir', default='../data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epsilon', type=float, default=8/255)
    parser.add_argument('--version', default='standard', choices=['standard', 'plus', 'rand'])
    parser.add_argument('--sub-depth', type=int, default=28, choices=[28, 34])
    parser.add_argument('--sub-widen', type=int, default=8, choices=[4, 8, 10])
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
        saved_args = ckpt.get("args", {})
        sub_depth = saved_args.get("sub_depth", args.sub_depth)
        sub_widen = saved_args.get("sub_widen", args.sub_widen)
    else:
        model_state = ckpt
        group_order = GROUP_ORDER
        coarse_groups = COARSE_GROUPS
        sub_depth = args.sub_depth
        sub_widen = args.sub_widen

    group_coarse = [coarse_groups[n] for n in group_order]
    group_fine = [build_fine_classes_for_group(g) for g in group_coarse]

    submodels = []
    for fine_classes in group_fine:
        m = WRNWithEmbedding(depth=sub_depth, widen_factor=sub_widen,
                             num_classes=len(fine_classes)).to(device)
        submodels.append(m)

    fusion = ParallelFusionWRN100(submodels, num_classes=100).to(device)
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
        args.log_path = os.path.join(model_dir, 'log-aa-parallel-fusion.log')

    try:
        from autoattack import AutoAttack
    except ImportError:
        raise ImportError('pip install autoattack')

    adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, log_path=args.log_path,
                           version=args.version, seed=args.seed)

    print(f'Model: {args.model_path}')
    print(f'Architecture: ParallelFusionWRN100 (4x WRN-{sub_depth}-{sub_widen})')
    print(f'Epsilon: {args.epsilon:.5f} ({args.epsilon*255:.2f}/255)')
    print(f'Version: {args.version}')
    print(f'Samples: {len(x_test)}')

    t0 = time.time()
    adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
    print(f'Completed in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
