"""
PGD White-box Attack for SharedWRN20Gate with unknown subset sampling.
Supports both with_unknown and no_unknown models.
Loads checkpoint_latest.pt format or standalone model files.
Standard autograd-based PGD implementation (no .data, no Variable).
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from models.wideresnet_update import BasicBlock, NetworkBlock


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
# Shared WRN (20 heads + unknown/no-unknown) for eval
# =========================================================
class SharedWRN20Gate(nn.Module):
    def __init__(self, depth=34, widen_factor=10, num_heads=20, num_known=5, unknown_idx=5,
                 gate_hidden=128, dropRate=0.0, fine_groups=None, n_block3=2, use_unknown=True):
        super().__init__()
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.num_heads = num_heads
        self.num_known = num_known
        self.unknown_idx = unknown_idx
        self.use_unknown = use_unknown

        # Shared backbone
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], BasicBlock, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], BasicBlock, 2, dropRate)

        # Heads with reduced block3
        self.block3 = nn.ModuleList([
            NetworkBlock(n_block3, nChannels[2], nChannels[3], BasicBlock, 2, dropRate)
            for _ in range(num_heads)
        ])
        self.bn = nn.ModuleList([nn.BatchNorm2d(nChannels[3]) for _ in range(num_heads)])
        out_dim = num_known + (1 if self.use_unknown else 0)
        self.fc = nn.ModuleList([nn.Linear(nChannels[3], out_dim) for _ in range(num_heads)])

        # Gate on shared features
        self.gate_fc1 = nn.Linear(nChannels[2], gate_hidden)
        self.gate_fc2 = nn.Linear(gate_hidden, num_heads)

        # Register fine indices for fusion
        if fine_groups is None:
            fine_groups = [[] for _ in range(num_heads)]
        for i, g in enumerate(fine_groups):
            self.register_buffer(f"fine_idx_{i}", torch.tensor(g, dtype=torch.long))

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / k) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _gate(self, shared_feat):
        pooled = nn.functional.adaptive_avg_pool2d(shared_feat, 1).view(shared_feat.size(0), -1)
        return self.gate_fc2(nn.functional.relu(self.gate_fc1(pooled)))

    def _head_forward(self, shared_feat, i):
        out = self.block3[i](shared_feat)
        out = self.relu(self.bn[i](out))
        out = nn.functional.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return self.fc[i](out)

    def forward(self, x, return_aux=False, return_gate=False, head=None):
        shared = self.conv1(x)
        shared = self.block1(shared)
        shared = self.block2(shared)

        gate_logits = self._gate(shared)  # (N, 20)

        if head is not None:
            return self._head_forward(shared, head)

        # Fusion to 100-way logits
        gate_log_probs = nn.functional.log_softmax(gate_logits, dim=1)
        batch_size = x.size(0)
        final_logits = x.new_full((batch_size, 100), -1e9)
        aux_logits = []
        for i in range(self.num_heads):
            logits_i = self._head_forward(shared, i)
            aux_logits.append(logits_i)
            fine_idx = getattr(self, f"fine_idx_{i}")
            log_g_i = gate_log_probs[:, i:i + 1]
            known_logits = logits_i[:, :len(fine_idx)]
            if self.use_unknown:
                unk_logit = logits_i[:, self.unknown_idx:self.unknown_idx + 1]
                final_logits[:, fine_idx] = log_g_i + (known_logits - unk_logit)
            else:
                final_logits[:, fine_idx] = log_g_i + known_logits

        if return_aux and return_gate:
            return aux_logits, final_logits, gate_logits
        if return_aux:
            return aux_logits, final_logits
        if return_gate:
            return final_logits, gate_logits
        return final_logits


def pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random_start=True):
    """PGD attack in pixel space [0,1]. Input X should be in [0,1].
    Normalization is applied dynamically before each model forward.
    Uses standard autograd-based PGD (no .data, no Variable, no per-step optimizer).
    """
    model.eval()
    
    # Prepare normalization tensors
    mean = torch.tensor(CIFAR100_MEAN, device=X.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=X.device).view(1, 3, 1, 1)
    
    def normalize(z):
        """Apply CIFAR-100 normalization."""
        return (z - mean) / std
    
    # Evaluate clean accuracy in pixel space
    with torch.no_grad():
        err = (model(normalize(X)).argmax(1) != y).float().sum().item()

    # Initialize adversarial input in pixel space [0,1]
    X0 = X.detach()
    if random_start:
        X_adv = (X0 + torch.empty_like(X0).uniform_(-epsilon, epsilon)).clamp(0, 1)
    else:
        X_adv = X0.clone()

    # PGD steps (standard autograd-based approach)
    for _ in range(num_steps):
        X_adv = X_adv.detach().requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(normalize(X_adv)), y)
        grad = torch.autograd.grad(loss, X_adv)[0]
        
        # Update in pixel space
        X_adv = X_adv + step_size * grad.sign()
        # Project back to epsilon ball
        delta = (X_adv - X0).clamp(-epsilon, epsilon)
        X_adv = (X0 + delta).clamp(0, 1)

    # Evaluate robust accuracy
    with torch.no_grad():
        err_pgd = (model(normalize(X_adv)).argmax(1) != y).float().sum().item()
    
    return err, err_pgd


def eval_adv_test_whitebox(model, loader, epsilon, num_steps, step_size, random_start=True):
    natural_err_total = 0
    robust_err_total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        err_nat, err_adv = pgd_whitebox(
            model, x, y,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            random_start=random_start
        )
        natural_err_total += err_nat
        robust_err_total += err_adv
    natural_acc = 1.0 - natural_err_total / len(loader.dataset)
    robust_acc = 1.0 - robust_err_total / len(loader.dataset)
    print('natural_acc: {:.4f}, robust_acc: {:.4f}'.format(natural_acc, robust_acc))


parser = argparse.ArgumentParser(description='PGD White-box Attack (CIFAR-100 Shared 20U with subset)')
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--step-size', type=float, default=0.003)
parser.add_argument('--random', action='store_true')
parser.add_argument('--no-random', dest='random', action='store_false')
parser.set_defaults(random=True)
parser.add_argument('--model-path', required=True, help='Path to model checkpoint or best_ema_model.pt')
parser.add_argument('--ema', action='store_true',
                    help='evaluate using EMA shadow in checkpoint')
parser.add_argument('--n-block3', type=int, default=2, help='Number of BasicBlocks in block3')
parser.add_argument('--use-unknown', action='store_true', default=True)
parser.add_argument('--no-unknown', dest='use_unknown', action='store_false')

args = parser.parse_args()
use_cuda = (not args.no_cuda) and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


def main():
    # data (do NOT normalize here; we'll normalize dynamically in pgd_whitebox)
    # This keeps input in pixel space [0, 1] for proper PGD clamping
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR100(
        root='../data', train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    # coarse->fine groups (20 experts, 5 fine classes each)
    coarse_to_fine = [[] for _ in range(20)]
    for fine_id, coarse_id in enumerate(CIFAR100_FINE_TO_COARSE):
        coarse_to_fine[coarse_id].append(fine_id)

    # Create model with correct use_unknown setting
    model = SharedWRN20Gate(
        depth=34, widen_factor=10, num_heads=20,
        num_known=5, unknown_idx=5, fine_groups=coarse_to_fine,
        n_block3=args.n_block3, use_unknown=args.use_unknown
    ).to(device)

    # Load checkpoint
    print(f'[INFO] Loading model from {args.model_path}')
    ckpt = torch.load(args.model_path, map_location=device)
    
    # Support both formats:
    # 1. Full checkpoint with 'model_state' and 'ema_state'
    # 2. Standalone model state dict
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        print('[INFO] Loaded model_state from checkpoint')
        
        if args.ema and 'ema_state' in ckpt:
            ema_state = ckpt['ema_state']
            for n, p in model.named_parameters():
                if n in ema_state:
                    p.data.copy_(ema_state[n].to(device))
            print('[INFO] Applied EMA shadow from checkpoint')
    else:
        # Assume it's a standalone model state dict
        model.load_state_dict(ckpt)
        print('[INFO] Loaded standalone model state dict')

    print(f'[INFO] Model use_unknown={model.use_unknown}')
    eval_adv_test_whitebox(
        model, test_loader,
        epsilon=args.epsilon,
        num_steps=args.num_steps,
        step_size=args.step_size,
        random_start=args.random
    )


if __name__ == '__main__':
    main()
