"""
AutoAttack evaluation for CIFAR-100 Gated Fusion (2 groups).
Usage: python auto_attack/eval_autoattack_cifar100_gated_2group.py \
         --model-path ./model-cifar100-gated-3stage-2group-wrn22-8/fusion-epoch100.pt
"""
from __future__ import print_function
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import torch
import torch.nn as nn
import torchvision
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
# Normalization wrapper for AutoAttack (expects [0,1] input)
# =========================================================
class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


# =========================================================
# Arguments
# =========================================================
parser = argparse.ArgumentParser(description='AutoAttack eval (CIFAR-100 Gated 2-group)')
parser.add_argument('--model-path', required=True)
parser.add_argument('--data-dir', default='../data')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epsilon', type=float, default=8/255)
parser.add_argument('--version', default='standard', choices=['standard', 'plus', 'rand'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--log-path', default=None)
parser.add_argument('--sub-depth', type=int, default=22)
parser.add_argument('--sub-widen', type=int, default=8)
parser.add_argument('--arch', default=None, choices=['wrn34-10', 'wrn22-8', 'wrn28-8'])
args = parser.parse_args()

ARCH_PRESETS = {'wrn34-10': (34, 10), 'wrn22-8': (22, 8), 'wrn28-8': (28, 8)}
if args.arch:
    args.sub_depth, args.sub_widen = ARCH_PRESETS[args.arch]

device = torch.device('cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu')
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)

# =========================================================
# Data (raw [0,1] for AutoAttack)
# =========================================================
transform_test = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

x_test = torch.cat([x for x, _ in test_loader], dim=0).to(device)
y_test = torch.cat([y for _, y in test_loader], dim=0).to(device)

# =========================================================
# Model
# =========================================================
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

# Wrap with normalization for AutoAttack
model_wrapped = NormalizedModel(model, CIFAR100_MEAN, CIFAR100_STD).to(device)
model_wrapped.eval()

# Log path
if args.log_path is None:
    model_dir = os.path.dirname(args.model_path)
    args.log_path = os.path.join(model_dir, 'log-aa-cifar100-gated-2group.log')

# =========================================================
# AutoAttack
# =========================================================
try:
    from autoattack import AutoAttack
except ImportError:
    raise ImportError('Please install autoattack: pip install autoattack')

adversary = AutoAttack(
    model_wrapped,
    norm='Linf',
    eps=args.epsilon,
    log_path=args.log_path,
    version=args.version,
    seed=args.seed
)

print('Running AutoAttack evaluation on CIFAR-100 (Gated 2-group)...')
print(f'  Model: {args.model_path}')
print(f'  Arch: WRN-{args.sub_depth}-{args.sub_widen} x2')
print(f'  Nature classes: {len(fine_nature)}, Manmade classes: {len(fine_manmade)}')
print(f'  Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)')
print(f'  Version: {args.version}')
print(f'  Samples: {len(x_test)}')

t0 = time.time()
x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
elapsed = time.time() - t0
print(f'AutoAttack completed in {elapsed:.1f}s')
print('Script completed.')
