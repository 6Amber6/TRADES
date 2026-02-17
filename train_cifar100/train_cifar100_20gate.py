from __future__ import print_function
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torchvision import transforms

from models.parallel_wrn import WRNWithEmbedding
from trades import trades_loss


# =========================================================
# CIFAR-100 Normalize (unused when training in pixel space)
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
# EMA
# =========================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])


# =========================================================
# Dataset wrapper for fine labels
# =========================================================
class CIFARFineSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, keep_fine_classes):
        self.base = base_dataset
        self.map = {c: i for i, c in enumerate(keep_fine_classes)}
        targets = torch.tensor(self.base.targets)
        self.idx = (
            (targets.unsqueeze(1) == torch.tensor(keep_fine_classes))
            .any(dim=1)
            .nonzero(as_tuple=False)
            .view(-1)
        )

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, y = self.base[self.idx[i]]
        return x, self.map[int(y)]


# =========================================================
# Backbone LR ratio schedule (Stage 2)
# =========================================================
def backbone_lr_ratio(epoch, total_epochs, r1=0.3, r2=0.3, r3=0.3):
    return r1


# =========================================================
# Soft routing fusion (20 experts + coarse gate)
# =========================================================
def _extract_logits(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class SoftGateFusionWRN100(nn.Module):
    def __init__(self, experts, fine_groups, gate):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        for i, g in enumerate(fine_groups):
            self.register_buffer(f"fine_idx_{i}", torch.tensor(g, dtype=torch.long))
        self.force_uniform_gate = False

    def forward(self, x, return_aux=False):
        if self.force_uniform_gate:
            gate_logits = torch.zeros(x.size(0), len(self.experts), device=x.device)
        else:
            gate_logits = _extract_logits(self.gate(x))
        gate_log_probs = F.log_softmax(gate_logits, dim=1)

        batch_size = x.size(0)
        device = x.device
        final_logits = torch.full((batch_size, 100), -1e9, device=device)
        expert_logits = []

        for i, expert in enumerate(self.experts):
            logits_i = _extract_logits(expert(x))
            expert_logits.append(logits_i)
            log_g_i = gate_log_probs[:, i:i + 1]
            fine_idx = getattr(self, f"fine_idx_{i}")
            final_logits[:, fine_idx] = log_g_i + logits_i
        if return_aux:
            return expert_logits, gate_logits, final_logits
        return final_logits


def aux_ce_loss(expert_logits, y, group_maps, weight=0.1):
    loss = 0.0
    for i, logits in enumerate(expert_logits):
        if logits is None:
            continue
        map_t = group_maps[i]
        y_local = map_t[y]
        mask = y_local >= 0
        if mask.any():
            loss = loss + F.cross_entropy(logits[mask], y_local[mask])
    return weight * loss


# =========================================================
# Arguments
# =========================================================
parser = argparse.ArgumentParser(description='TRADES CIFAR-100 20-expert soft routing (WRN-16-4)')
parser.add_argument('--epochs-sub', type=int, default=100)
parser.add_argument('--epochs-fusion', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=10)
parser.add_argument('--step-size', type=float, default=0.007)
parser.add_argument('--beta', type=float, default=6.0)
parser.add_argument('--model-dir', default='./model-cifar100-20gate')
parser.add_argument('--data-dir', default='../data')
parser.add_argument('--resume', default='auto',
                    help='checkpoint path or "auto" to resume from model-dir/checkpoint-last.pt')
parser.add_argument('--aux-weight', type=float, default=0.1)
parser.add_argument('--gate-weight', type=float, default=0.4)
parser.add_argument('--gate-warmup', type=int, default=10,
                    help='freeze gate for first N fusion epochs')
parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'none'])
parser.add_argument('--seed', type=int, default=42)


def train_ce_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = _extract_logits(model(x))
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def main():
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =========================================================
    # Build coarse -> fine mapping (20 experts, 5 fine classes each)
    # =========================================================
    coarse_to_fine = [[] for _ in range(20)]
    for fine_id, coarse_id in enumerate(CIFAR100_FINE_TO_COARSE):
        coarse_to_fine[coarse_id].append(fine_id)

    # Precompute expert label maps for aux loss
    group_maps = []
    for fine_ids in coarse_to_fine:
        map_t = torch.full((100,), -1, dtype=torch.long, device=device)
        map_t[torch.tensor(fine_ids, device=device)] = torch.arange(len(fine_ids), device=device)
        group_maps.append(map_t)
    fine_to_coarse_t = torch.tensor(CIFAR100_FINE_TO_COARSE, device=device, dtype=torch.long)

    # =========================================================
    # Data
    # =========================================================
    transform_sub = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.05, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
    ])
    transform_fusion = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandAugment(num_ops=2, magnitude=4),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    base_train_sub = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True, transform=transform_sub
    )
    base_train_fusion = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True, transform=transform_fusion
    )
    test_set = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )
    train_loader_100 = torch.utils.data.DataLoader(
        base_train_fusion,
        batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )

    # =========================================================
    # Stage 1: pretrain 20 experts (WRN-16-4)
    # =========================================================
    resume_path = args.resume
    if resume_path == 'auto':
        resume_path = os.path.join(args.model_dir, 'checkpoint-last.pt')

    resume_loaded = False
    start_epoch = 1
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(f'[INFO] Resuming from {resume_path}')
            resume_loaded = True
            start_epoch = checkpoint.get('epoch', 0) + 1

    if not resume_loaded:
        print('==== Stage 1: Pretrain 20 experts (WRN-16-4) ====')
        for i, fine_ids in enumerate(coarse_to_fine):
            expert = WRNWithEmbedding(depth=16, widen_factor=4, num_classes=len(fine_ids)).to(device)
            loader = torch.utils.data.DataLoader(
                CIFARFineSubset(base_train_sub, fine_ids),
                batch_size=args.batch_size, shuffle=True,
                num_workers=8, pin_memory=True,
                persistent_workers=True, prefetch_factor=4
            )
            opt = optim.SGD(expert.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
            for ep in range(1, args.epochs_sub + 1):
                l, a = train_ce_epoch(expert, loader, opt, device)
                if ep % 10 == 0 or ep == 1 or ep == args.epochs_sub:
                    print(f'[Expert {i:02d}][{ep}/{args.epochs_sub}] loss={l:.4f} acc={a*100:.2f}%')
            torch.save(expert.state_dict(), os.path.join(args.model_dir, f'expert_{i:02d}_final.pt'))
            del expert
            torch.cuda.empty_cache()

    # =========================================================
    # Stage 2: soft routing fusion with coarse gate
    # =========================================================
    experts = []
    for i, fine_ids in enumerate(coarse_to_fine):
        m = WRNWithEmbedding(depth=16, widen_factor=4, num_classes=len(fine_ids)).to(device)
        ckpt_path = os.path.join(args.model_dir, f'expert_{i:02d}_final.pt')
        if os.path.isfile(ckpt_path):
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
        experts.append(m)

    gate = WRNWithEmbedding(depth=16, widen_factor=4, num_classes=20).to(device)
    fusion = SoftGateFusionWRN100(experts, coarse_to_fine, gate).to(device)

    ema = EMA(fusion, decay=0.999)
    if resume_loaded and isinstance(checkpoint, dict):
        fusion.load_state_dict(checkpoint['model_state_dict'])
        loaded_shadow = checkpoint.get('ema_shadow')
        if loaded_shadow is not None:
            ema.shadow = {k: v.to(device) for k, v in loaded_shadow.items()}

    # -------- Gate warmup (train gate only) --------
    if (not resume_loaded) and args.gate_warmup > 0:
        print(f'==== Gate warmup ({args.gate_warmup} epochs) ====')
        gate_optimizer = optim.SGD(
            fusion.gate.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )
        for m in fusion.experts:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
        fusion.train()
        for ep in range(1, args.gate_warmup + 1):
            for x, y in train_loader_100:
                x, y = x.to(device), y.to(device)
                gate_optimizer.zero_grad()
                gate_logits = _extract_logits(fusion.gate(x))
                coarse_y = fine_to_coarse_t[y]
                loss = args.gate_weight * F.cross_entropy(gate_logits, coarse_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion.gate.parameters(), 5.0)
                gate_optimizer.step()
                ema.update(fusion)
            print(f'[GateWarmup] Epoch {ep}/{args.gate_warmup}, lr={gate_optimizer.param_groups[0]["lr"]:.6f}')

    # Build optimizer for joint training
    params = [{'params': fusion.gate.parameters(), 'lr': args.lr}]
    for m in fusion.experts:
        for p in m.parameters():
            p.requires_grad = True
        m.train()
        params.append({'params': m.parameters(), 'lr': args.lr})

    optimizer = optim.SGD(
        params,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    if resume_loaded and isinstance(checkpoint, dict):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # -------- TRADES training --------
    print('==== TRADES training ====')
    if args.scheduler == 'step':
        milestones = [int(args.epochs_fusion * 0.5), int(args.epochs_fusion * 0.75)]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_fusion)
    else:
        scheduler = None

    if resume_loaded and isinstance(checkpoint, dict):
        sched_state = checkpoint.get('scheduler_state_dict')
        if sched_state is not None and scheduler is not None:
            scheduler.load_state_dict(sched_state)

    if (not resume_loaded) and args.gate_warmup > 0:
        start_epoch = max(start_epoch, args.gate_warmup + 1)

    for ep in range(start_epoch, args.epochs_fusion + 1):
        fusion.train()
        total_loss = 0.0
        total_samples = 0

        ratio = backbone_lr_ratio(ep, args.epochs_fusion)
        base_lr = optimizer.param_groups[0]['lr']
        # experts: group 1..20
        for g in range(1, len(optimizer.param_groups)):
            optimizer.param_groups[g]['lr'] = base_lr * ratio
        fusion.force_uniform_gate = False

        for x, y in train_loader_100:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = trades_loss(
                model=fusion,
                x_natural=x,
                y=y,
                optimizer=optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta,
                data_mean=None,
                data_std=None
            )

            expert_logits, gate_logits, _ = fusion(x, return_aux=True)
            loss = loss + aux_ce_loss(expert_logits, y, group_maps, weight=args.aux_weight)
            coarse_y = fine_to_coarse_t[y]
            loss = loss + args.gate_weight * F.cross_entropy(gate_logits, coarse_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        if scheduler is not None:
            scheduler.step()

        acc = None
        if ep % 5 == 0 or ep == args.epochs_fusion:
            # EMA evaluation (every 5 epochs)
            test_correct, test_total = 0, 0
            fusion.eval()
            with torch.no_grad():
                ema.apply_to(fusion)
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    pred = fusion(x).argmax(1)
                    test_correct += (pred == y).sum().item()
                    test_total += y.size(0)
                ema.restore(fusion)
            fusion.train()
            acc = test_correct / max(1, test_total)
        avg_loss = total_loss / max(1, total_samples)
        if acc is None:
            print(f'[Fusion/TRADES][{ep}/{args.epochs_fusion}] test_acc=skip, train_loss={avg_loss:.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')
        else:
            print(f'[Fusion/TRADES][{ep}/{args.epochs_fusion}] test_acc={acc*100:.2f}%, train_loss={avg_loss:.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')

        if ep >= 40 and ep % 5 == 0:
            torch.save(
                {
                    'epoch': ep,
                    'model_state_dict': fusion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'ema_shadow': ema.shadow,
                    'args': vars(args),
                },
                os.path.join(args.model_dir, 'checkpoint-last.pt')
            )


if __name__ == '__main__':
    main()
