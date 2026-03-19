"""
AutoAttack evaluation for GatedFusionWRN (train_trades_cifar10_parallel_warmup_ema_aux_lrstage_gated.py).

Run from repo root:
  python auto_attack/eval_autoattack_cifar10_gated.py \
      --model-path ./train_cifar10/model-parallel-gated/fusion-epoch100.pt

Checkpoint formats accepted:
  - fusion-epoch{N}.pt  : raw state_dict
  - checkpoint-last.pt  : dict with 'model_state_dict' key
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

# ---- constants ----
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
VEHICLE_CLASSES = [0, 1, 8, 9]
ANIMAL_CLASSES  = [2, 3, 4, 5, 6, 7]


class GatedFusionWRN(nn.Module):
    """Must match exactly the definition in train_trades_cifar10_parallel_warmup_ema_aux_lrstage_gated.py."""
    def __init__(self, m4, m6, emb_dim=640, hidden_dim=256):
        super().__init__()
        self.m4 = m4
        self.m6 = m6
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Linear(emb_dim, 10)

    def forward(self, x, return_aux=False):
        e4, out4 = self.m4(x, return_embedding=True)
        e6, out6 = self.m6(x, return_embedding=True)
        gate_in = torch.cat([e4, e6], dim=1)
        alpha = torch.sigmoid(self.gate(gate_in))
        emb = alpha * e4 + (1 - alpha) * e6
        out = self.fc(emb)
        if return_aux:
            return out4, out6, out
        return out


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
        description='AutoAttack eval – GatedFusionWRN')
    parser.add_argument('--model-path', required=True,
                        help='path to fusion checkpoint')
    parser.add_argument('--data-dir',   default='../data')
    parser.add_argument('--batch-size', type=int,   default=128)
    parser.add_argument('--epsilon',    type=float, default=8/255,
                        help='Linf epsilon (default: 8/255)')
    parser.add_argument('--version',    default='standard',
                        choices=['standard', 'plus', 'rand'])
    parser.add_argument('--sub-depth',  type=int,   default=34,
                        help='WRN depth (34=WRN-34-10, 16=WRN-16-8)')
    parser.add_argument('--sub-widen',  type=int,   default=10,
                        help='WRN widen factor (10=WRN-34-10, 8=WRN-16-8)')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--no-cuda',    action='store_true')
    parser.add_argument('--log-path',   default=None)
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
    emb_dim = 64 * args.sub_widen
    m4 = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=4).to(device)
    m6 = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=6).to(device)
    fusion = GatedFusionWRN(m4, m6, emb_dim=emb_dim).to(device)

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
        args.log_path = os.path.join(model_dir, 'log-aa-gated.log')

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
    print(f'Samples: {len(x_test)}')

    t0 = time.time()
    adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
    print(f'Completed in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
