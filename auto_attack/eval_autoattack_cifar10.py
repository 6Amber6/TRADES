"""
AutoAttack evaluation for TRADES CIFAR-10 model.
Usage: pip install autoattack
       python auto_attack/eval_autoattack_cifar10.py --model-path ./model-cifar-wideResNet/model-wideres-epoch76.pt
"""

from __future__ import print_function
import os
import argparse
import time
import torch
import torchvision
from torchvision import transforms

from models.wideresnet import WideResNet

parser = argparse.ArgumentParser(description='AutoAttack evaluation for TRADES CIFAR-10')
parser.add_argument('--model-path', required=True, help='path to model checkpoint')
parser.add_argument('--data-dir', default='../data', help='CIFAR-10 data directory')
parser.add_argument('--batch-size', type=int, default=128, help='batch size for evaluation')
parser.add_argument('--epsilon', type=float, default=8/255, help='Linf epsilon (default: 8/255)')
parser.add_argument('--version', default='standard', choices=['standard', 'plus', 'rand'],
                    help='AutoAttack version')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--log-path', default=None, help='log file path (default: model_dir/log-aa.log)')
args = parser.parse_args()

device = torch.device('cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu')
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)

# Data (no normalization - TRADES CIFAR-10 uses raw [0,1])
transform_test = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Concatenate all test data for AutoAttack
x_test = torch.cat([x for x, _ in test_loader], dim=0)
y_test = torch.cat([y for _, y in test_loader], dim=0)
x_test = x_test.to(device)
y_test = y_test.to(device)

# Model
model = WideResNet(depth=34, num_classes=10, widen_factor=10).to(device)
ckpt = torch.load(args.model_path, map_location=device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model.eval()

# Log path
if args.log_path is None:
    model_dir = os.path.dirname(args.model_path)
    args.log_path = os.path.join(model_dir, 'log-aa-cifar10.log')

# AutoAttack
try:
    from autoattack import AutoAttack
except ImportError:
    raise ImportError('Please install autoattack: pip install autoattack')

adversary = AutoAttack(
    model,
    norm='Linf',
    eps=args.epsilon,
    log_path=args.log_path,
    version=args.version,
    seed=args.seed
)

print('Running AutoAttack evaluation on CIFAR-10...')
print('  Model: {}'.format(args.model_path))
print('  Epsilon: {:.4f} ({}/255)'.format(args.epsilon, args.epsilon*255))
print('  Version: {}'.format(args.version))
print('  Samples: {}'.format(len(x_test)))

t0 = time.time()
x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
elapsed = time.time() - t0
print('AutoAttack completed in {:.1f}s'.format(elapsed))
print('Script completed.')
