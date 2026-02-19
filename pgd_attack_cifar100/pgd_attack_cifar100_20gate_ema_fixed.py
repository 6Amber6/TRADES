"""
PGD White-box Attack for SoftGateFusionWRN100 (20-expert soft routing).
Loads checkpoint-last.pt format with EMA support.
Standard autograd-based PGD implementation (no .data, no Variable, no per-step optimizer).
Dynamic normalization applied during PGD in pixel space.
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# =========================================================
# Utils
# =========================================================
def _extract_logits(out):
    """Extract logits if output is tuple/list, otherwise return as-is."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class SoftGateFusionWRN100(nn.Module):
    """20-expert soft routing fusion model."""
    def __init__(self, experts, fine_groups, gate):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        for i, g in enumerate(fine_groups):
            self.register_buffer(f"fine_idx_{i}", torch.tensor(g, dtype=torch.long))

    def forward(self, x):
        gate_logits = _extract_logits(self.gate(x))
        gate_log_probs = F.log_softmax(gate_logits, dim=1)
        batch_size = x.size(0)
        device = x.device
        final_logits = torch.full((batch_size, 100), -1e9, device=device)
        for i, expert in enumerate(self.experts):
            logits_i = _extract_logits(expert(x))
            log_g_i = gate_log_probs[:, i:i + 1]
            fine_idx = getattr(self, f"fine_idx_{i}")
            final_logits[:, fine_idx] = log_g_i + logits_i
        return final_logits


def pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random_start=True):
    """
    PGD attack in pixel space [0,1].
    Input X should be in [0,1].
    Model is trained on pixel-space (no normalization).
    Uses standard autograd-based PGD (no .data, no Variable, no per-step optimizer).
    """
    model.eval()
    
    # Evaluate clean accuracy in pixel space (no normalization)
    with torch.no_grad():
        err = (model(X).argmax(1) != y).float().sum().item()

    # Initialize adversarial input in pixel space [0,1]
    X0 = X.detach()
    if random_start:
        X_adv = (X0 + torch.empty_like(X0).uniform_(-epsilon, epsilon)).clamp(0, 1)
    else:
        X_adv = X0.clone()

    # PGD steps (standard autograd-based approach)
    for _ in range(num_steps):
        X_adv = X_adv.detach().requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(X_adv), y)
        grad = torch.autograd.grad(loss, X_adv)[0]
        
        # Update in pixel space
        X_adv = X_adv + step_size * grad.sign()
        # Project back to epsilon ball
        delta = (X_adv - X0).clamp(-epsilon, epsilon)
        X_adv = (X0 + delta).clamp(0, 1)

    # Evaluate robust accuracy
    with torch.no_grad():
        err_pgd = (model(X_adv).argmax(1) != y).float().sum().item()
    
    return err, err_pgd


def eval_adv_test_whitebox(model, loader, epsilon, num_steps, step_size, random_start=True):
    """Evaluate natural and robust accuracy."""
    natural_err_total = 0
    robust_err_total = 0
    batch_count = 0
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
        batch_count += 1
        print(f'  [Batch {batch_count}] nat_err={err_nat}, pgd_err={err_adv}')
    
    natural_acc = 1.0 - natural_err_total / len(loader.dataset)
    robust_acc = 1.0 - robust_err_total / len(loader.dataset)
    print(f'natural_acc: {natural_acc:.4f}, robust_acc: {robust_acc:.4f}')


parser = argparse.ArgumentParser(description='PGD White-box Attack (CIFAR-100 20-gate soft routing)')
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--step-size', type=float, default=0.003)
parser.add_argument('--random', action='store_true')
parser.add_argument('--no-random', dest='random', action='store_false')
parser.set_defaults(random=True)
parser.add_argument('--model-path', required=True, help='Path to checkpoint-last.pt')
parser.add_argument('--ema', action='store_true',
                    help='Apply EMA shadow from checkpoint')

args = parser.parse_args()
use_cuda = (not args.no_cuda) and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


def main():
    # Data (do NOT normalize here; we'll normalize dynamically in pgd_whitebox)
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

    # Build coarse -> fine groups (20 experts, 5 fine classes each)
    coarse_to_fine = [[] for _ in range(20)]
    for fine_id, coarse_id in enumerate(CIFAR100_FINE_TO_COARSE):
        coarse_to_fine[coarse_id].append(fine_id)

    # Reconstruct experts and gate from checkpoint
    print(f'[INFO] Loading checkpoint from {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        print('[ERROR] Checkpoint does not contain expected format (model_state_dict)')
        return

    # Create experts
    experts = []
    for i, fine_ids in enumerate(coarse_to_fine):
        m = WRNWithEmbedding(depth=16, widen_factor=4, num_classes=len(fine_ids)).to(device)
        experts.append(m)

    # Create gate
    gate = WRNWithEmbedding(depth=16, widen_factor=4, num_classes=20).to(device)
    
    # Create fusion model
    fusion = SoftGateFusionWRN100(experts, coarse_to_fine, gate).to(device)
    
    # Load model state
    fusion.load_state_dict(checkpoint['model_state_dict'])
    print('[INFO] Loaded model_state_dict from checkpoint')

    # Apply EMA if requested
    if args.ema:
        ema_shadow = checkpoint.get('ema_shadow')
        if ema_shadow is not None:
            for n, p in fusion.named_parameters():
                if n in ema_shadow:
                    p.data.copy_(ema_shadow[n].to(device))
            print('[INFO] Applied EMA shadow from checkpoint')
        else:
            print('[WARNING] --ema requested but no ema_shadow in checkpoint')

    fusion.eval()
    eval_adv_test_whitebox(
        fusion, test_loader,
        epsilon=args.epsilon,
        num_steps=args.num_steps,
        step_size=args.step_size,
        random_start=args.random
    )


if __name__ == '__main__':
    main()
