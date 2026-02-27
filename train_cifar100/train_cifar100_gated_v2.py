# WideResNet 28-8 *4
# feature-level soft routing
# Gate v2: Linear -> BN -> ReLU -> Dropout -> Linear
# Freeze backbone first 40 ep, then unfreeze with r1=0.05

from __future__ import print_function
import os
import argparse
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
# Meta-groups by coarse class IDs (customize here)
# =========================================================
COARSE_GROUPS = {
    "textured_organic": [7, 15, 16, 8, 13],
    "smooth_organic": [1, 0, 11, 12, 14],
    "rigid_manmade": [6, 5, 3, 18, 9],
    "large_structures": [10, 17, 19, 2, 4],
}
GROUP_ORDER = ["textured_organic", "smooth_organic", "rigid_manmade", "large_structures"]


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
# BN freeze/unfreeze util
# =========================================================
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def unfreeze_bn(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.train()
            for p in m.parameters():
                p.requires_grad = True


# =========================================================
# Gated fusion v2: projection + dropout + BN (anti-overfit)
# - Project each 640-dim emb to 64-dim, then concat
# - Gate: Linear -> BN -> ReLU -> Dropout(0.5) -> Linear
# =========================================================
class GatedFusionWRN100(nn.Module):
    def __init__(self, submodels, num_classes=100, freeze_backbone=False, hidden_dim=256, proj_dim=64):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        if freeze_backbone:
            for m in self.submodels:
                for p in m.parameters():
                    p.requires_grad = False
        emb_dim = self.submodels[0].nChannels
        n_experts = len(self.submodels)

        # Per-expert projection: 640 -> 64
        self.proj = nn.ModuleList([
            nn.Linear(emb_dim, proj_dim) for _ in range(n_experts)
        ])
        gate_in_dim = proj_dim * n_experts

        # Gate: Linear -> BN -> ReLU -> Dropout -> Linear
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
        # Project each emb: 640 -> 64
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


# =========================================================
# Aux loss for 4 meta-groups
# =========================================================
def aux_ce_loss(group_logits, y, group_defs, fine_to_coarse_t, weight=0.1):
    loss = 0.0
    for i, g in enumerate(group_defs):
        logits = group_logits[i]
        if logits is None:
            continue
        coarse_y = fine_to_coarse_t[y]
        mask = torch.isin(coarse_y, g["coarse_t"])
        if mask.any():
            y_group = g["map_t"][y[mask]]
            loss += F.cross_entropy(logits[mask], y_group)
    return weight * loss


# =========================================================
# Arguments
# =========================================================
parser = argparse.ArgumentParser(description='TRADES Parallel WRN CIFAR-100 (4-group) Gated v2')
parser.add_argument('--epochs-sub', type=int, default=120)
parser.add_argument('--epochs-fusion', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=10)
parser.add_argument('--step-size', type=float, default=0.007)
parser.add_argument('--beta', type=float, default=6.0)
parser.add_argument('--sub-depth', type=int, default=28, choices=[28, 34])
parser.add_argument('--sub-widen', type=int, default=8, choices=[4, 8, 10])
parser.add_argument('--aux-weight', type=float, default=0.1)
parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'none'])
parser.add_argument('--model-dir', default='./model-cifar100-gated-v2')
parser.add_argument('--resume', default='auto')
parser.add_argument('--freeze-backbone-epochs', type=int, default=80,
                    help='freeze backbone for first N TRADES epochs, then unfreeze with r1')
parser.add_argument('--backbone-lr-ratio', type=float, default=0.01,
                    help='backbone lr = base_lr * ratio after unfreeze')
parser.add_argument('--backbone-lr-warmup-epochs', type=int, default=10,
                    help='linear warmup epochs for backbone lr after unfreeze')
args = parser.parse_args()


# =========================================================
# Training utilities
# =========================================================
def train_ce_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def build_fine_classes_for_group(group_coarse):
    return [i for i, c in enumerate(CIFAR100_FINE_TO_COARSE) if c in group_coarse]


# =========================================================
# Main
# =========================================================
def main():
    import random
    import numpy as np
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_dir, exist_ok=True)

    transform_sub = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        transforms.RandomErasing(p=0.05, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
    ])
    transform_fusion = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    base_train_sub = torchvision.datasets.CIFAR100(
        root='../data', train=True, download=True, transform=transform_sub
    )
    base_train_fusion = torchvision.datasets.CIFAR100(
        root='../data', train=True, download=True, transform=transform_fusion
    )
    test_set = torchvision.datasets.CIFAR100(
        root='../data', train=False, download=True, transform=transform_test
    )

    group_names = GROUP_ORDER
    group_coarse = [COARSE_GROUPS[name] for name in group_names]
    group_fine = [build_fine_classes_for_group(g) for g in group_coarse]

    group_loaders = []
    for fine_classes in group_fine:
        loader = torch.utils.data.DataLoader(
            CIFARFineSubset(base_train_sub, fine_classes),
            batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        group_loaders.append(loader)

    train_loader_100 = torch.utils.data.DataLoader(
        base_train_fusion,
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # ================= Resume =================
    resume_path = args.resume
    if resume_path == 'auto':
        resume_path = os.path.join(args.model_dir, 'checkpoint-last.pt')

    resume_loaded = False
    start_epoch = 1
    warmup_start = 1
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(f'[INFO] Resuming from {resume_path}')
            resume_loaded = True
            stage2_epoch = checkpoint.get('stage2_epoch', 0)
            if stage2_epoch <= 10:
                warmup_start = stage2_epoch + 1
                start_epoch = 1
            else:
                warmup_start = 11
                start_epoch = stage2_epoch - 10 + 1

    # ================= Stage 1 =================
    submodels = []
    for i in range(len(group_loaders)):
        m = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=len(group_fine[i])).to(device)
        submodels.append(m)

    if not resume_loaded:
        print('==== Stage 1 (CIFAR-100 fine, 4 groups) ====')
        for i, loader in enumerate(group_loaders):
            m = submodels[i]
            opt = optim.SGD(m.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for ep in range(1, args.epochs_sub + 1):
                _, a = train_ce_epoch(m, loader, opt, device)
                print(f'[Sub][{ep}] {group_names[i]} acc={a*100:.2f}%')
            torch.save(m.state_dict(), f'{args.model_dir}/wrn_{group_names[i]}_final.pt')
            m.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ================= Stage 2 =================
    for m in submodels:
        m.to(device)
        for p in m.parameters():
            p.requires_grad = True

    fusion = GatedFusionWRN100(submodels, num_classes=100, freeze_backbone=False).to(device)

    # Param groups: [0]=fc, [1..4]=submodels, [5]=gate
    params = [{'params': fusion.fc.parameters(), 'lr': args.lr}]
    for m in fusion.submodels:
        params.append({'params': m.parameters(), 'lr': args.lr})
    params.append({'params': list(fusion.proj.parameters()) + list(fusion.gate.parameters()), 'lr': args.lr})

    optimizer = optim.SGD(
        params,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    print("Trainable params:", sum(p.numel() for p in fusion.parameters() if p.requires_grad))
    ema = EMA(fusion, decay=0.999)

    if resume_loaded and checkpoint is not None:
        fusion.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            ckpt_opt = checkpoint['optimizer_state_dict']
            n_cur, n_ckpt = len(optimizer.param_groups), len(ckpt_opt.get('param_groups', []))
            if n_cur != n_ckpt:
                raise ValueError(f'[RESUME] optimizer param_groups mismatch: current={n_cur}, checkpoint={n_ckpt}')
            optimizer.load_state_dict(ckpt_opt)
            print(f'[RESUME] optimizer loaded: {n_cur} param_groups')
        if 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']

    fine_to_coarse_t = torch.tensor(CIFAR100_FINE_TO_COARSE, device=device, dtype=torch.long)
    group_defs = []
    for coarse_ids, fine_ids in zip(group_coarse, group_fine):
        coarse_t = torch.tensor(coarse_ids, device=device, dtype=torch.long)
        map_t = torch.full((100,), -1, device=device, dtype=torch.long)
        map_t[torch.tensor(fine_ids, device=device)] = torch.arange(len(fine_ids), device=device)
        group_defs.append({"coarse_t": coarse_t, "map_t": map_t})

    # -------- CE warmup (10 epochs) --------
    print('==== CE warmup (10 epochs) ====')
    for ep in range(warmup_start, 11):
        fusion.train()
        for x, y in train_loader_100:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = fusion(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)
        print(f'[Warmup] Epoch {ep}/10, lr={optimizer.param_groups[0]["lr"]:.6f}')

        ckpt = {
            'model_state_dict': fusion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_shadow': ema.shadow,
            'stage2_epoch': ep,
        }
        torch.save(ckpt, os.path.join(args.model_dir, 'checkpoint-last.pt'))

    # -------- TRADES training --------
    print('==== TRADES training ====')
    if args.scheduler == 'step':
        milestones = [int(args.epochs_fusion * 0.5), int(args.epochs_fusion * 0.75)]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_fusion)
    else:
        scheduler = None

    if resume_loaded and checkpoint is not None and scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    freeze_epochs = args.freeze_backbone_epochs
    r1 = args.backbone_lr_ratio
    warmup_epochs = args.backbone_lr_warmup_epochs

    for ep in range(start_epoch, args.epochs_fusion + 1):
        fusion.train()

        fusion_lr = optimizer.param_groups[0]['lr']
        if ep <= freeze_epochs:
            for m in fusion.submodels:
                for p in m.parameters():
                    p.requires_grad = False
                freeze_bn(m)
            backbone_lr = 0.0
            for g in range(1, 1 + len(fusion.submodels)):
                optimizer.param_groups[g]['lr'] = 0.0
        else:
            for m in fusion.submodels:
                for p in m.parameters():
                    p.requires_grad = True
                unfreeze_bn(m)
            if ep <= freeze_epochs + warmup_epochs:
                progress = (ep - freeze_epochs) / warmup_epochs
                backbone_lr = fusion_lr * r1 * progress
            else:
                backbone_lr = fusion_lr * r1
            for g in range(1, 1 + len(fusion.submodels)):
                optimizer.param_groups[g]['lr'] = backbone_lr

        total_loss = 0.0
        total_samples = 0

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
                data_mean=CIFAR100_MEAN,
                data_std=CIFAR100_STD
            )

            aux_logits, _ = fusion(x, return_aux=True)
            loss += aux_ce_loss(aux_logits, y, group_defs, fine_to_coarse_t, weight=args.aux_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        if scheduler is not None:
            scheduler.step()

        test_correct, test_total = 0, 0
        fusion.eval()
        with torch.no_grad():
            ema.apply_to(fusion)
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = fusion(x, return_aux=True)[-1].argmax(1)
                test_correct += (pred == y).sum().item()
                test_total += y.size(0)
            ema.restore(fusion)
        fusion.train()

        acc = test_correct / test_total
        avg_loss = total_loss / max(1, total_samples)
        freeze_info = 'backbone_frozen' if ep <= freeze_epochs else f'backbone_lr={backbone_lr:.6f}'
        print(f'[Fusion/TRADES][{ep}/{args.epochs_fusion}] test_acc={acc*100:.2f}%, train_loss={avg_loss:.4f}, {freeze_info}')

        ckpt = {
            'model_state_dict': fusion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'ema_shadow': ema.shadow,
            'stage2_epoch': 10 + ep,
        }
        torch.save(ckpt, os.path.join(args.model_dir, 'checkpoint-last.pt'))

        if ep % 5 == 0 or ep == args.epochs_fusion:
            if ep == args.epochs_fusion:
                with torch.no_grad():
                    for n, p in fusion.named_parameters():
                        if n in ema.shadow:
                            p.data.copy_(ema.shadow[n])
                print('[INFO] Applied EMA weights to final checkpoint')
            torch.save(
                {
                    "state_dict": fusion.state_dict(),
                    "group_order": GROUP_ORDER,
                    "coarse_groups": COARSE_GROUPS,
                    "args": vars(args),
                },
                f'{args.model_dir}/fusion-epoch{ep}.pt'
            )


if __name__ == '__main__':
    main()
