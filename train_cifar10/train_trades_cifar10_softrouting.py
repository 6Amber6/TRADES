from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

from models.parallel_wrn import WRNWithEmbedding
from trades import trades_loss


# =========================================================
# CIFAR-10 class split
# =========================================================
VEHICLE_CLASSES = [0, 1, 8, 9]
ANIMAL_CLASSES = [2, 3, 4, 5, 6, 7]


# =========================================================
# CIFAR-10 Normalize
# =========================================================
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


# =========================================================
# Soft routing helpers
# =========================================================
def map_expert_to_10(logits_local, owned_classes, num_classes=10, margin=0.0):
    """
    logits_local: [B, K+1]  (last column = unknown)
    """
    B = logits_local.size(0)
    K = len(owned_classes)

    known = logits_local[:, :K]       # [B,K]
    unk = logits_local[:, K:K+1]      # [B,1]

    logits_10 = (unk - margin).expand(B, num_classes).clone()
    for i, c in enumerate(owned_classes):
        logits_10[:, c] = known[:, i]

    return logits_10


def soft_routing_fusion(logits4_local, logits6_local, a=1.0, b=0.5, T=1.0, margin=0.0):
    """
    logits4_local: [B,5]  (4 classes + unk)
    logits6_local: [B,7]  (6 classes + unk)
    """
    p4 = F.softmax(logits4_local, dim=1)
    p6 = F.softmax(logits6_local, dim=1)

    unk4 = p4[:, -1:]  # [B,1]
    unk6 = p6[:, -1:]

    conf4 = 1 - unk4
    conf6 = 1 - unk6

    s4 = a * conf4 + b * unk6
    s6 = a * conf6 + b * unk4

    s = torch.cat([s4, s6], dim=1)  # [B,2]
    w = F.softmax(s / T, dim=1)     # [B,2]
    w4, w6 = w[:, :1], w[:, 1:]

    logits4_10 = map_expert_to_10(logits4_local, VEHICLE_CLASSES, margin=margin)
    logits6_10 = map_expert_to_10(logits6_local, ANIMAL_CLASSES, margin=margin)

    final_logits = w4 * logits4_10 + w6 * logits6_10

    return final_logits, w4, w6


class SoftRoutingFusion(nn.Module):
    def __init__(self, m4, m6, a=1.0, b=0.5, T=1.0, margin=0.0):
        super().__init__()
        self.m4 = m4
        self.m6 = m6
        self.a = a
        self.b = b
        self.T = T
        self.margin = margin

    def forward(self, x, return_aux=False):
        out4 = self.m4(x)
        out6 = self.m6(x)
        logits4 = out4[0] if isinstance(out4, (tuple, list)) else out4
        logits6 = out6[0] if isinstance(out6, (tuple, list)) else out6
        final_logits, _, _ = soft_routing_fusion(
            logits4, logits6, a=self.a, b=self.b, T=self.T, margin=self.margin
        )
        if return_aux:
            return logits4, logits6, final_logits
        return final_logits


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
# Dataset wrapper (known + unknown)
# =========================================================
class CIFARUnknownDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, known_classes):
        self.base = base_dataset
        self.known = list(known_classes)
        self.map = {c: i for i, c in enumerate(self.known)}
        self.known_set = set(self.known)
        self.unknown_idx = len(self.known)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if y in self.known_set:
            y = self.map[y]
        else:
            y = self.unknown_idx
        return x, y


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
# Aux loss (only known classes)
# =========================================================
def aux_ce_loss(logits4, logits6, y, lut4, lut6, weight=0.02):
    loss = 0.0
    if logits4 is not None:
        idx4 = lut4[y]
        m4 = idx4 != -1
        if m4.any():
            loss += F.cross_entropy(logits4[m4], idx4[m4])

    if logits6 is not None:
        idx6 = lut6[y]
        m6 = idx6 != -1
        if m6.any():
            loss += F.cross_entropy(logits6[m6], idx6[m6])

    return weight * loss


# =========================================================
# Backbone LR ratio schedule (Stage 2)
# =========================================================
def backbone_lr_ratio(epoch, total_epochs, r1=0.15, r2=0.5, r3=0.35):
    p1 = max(1, int(total_epochs * 0.3))
    p2 = max(p1 + 1, int(total_epochs * 0.7))
    if epoch <= p1:
        return r1
    if epoch <= p2:
        return r2
    return r3


# =========================================================
# Arguments
# =========================================================
parser = argparse.ArgumentParser(description='TRADES Parallel WRN Training (soft routing)')
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
parser.add_argument('--model-dir', default='./model-parallel')
parser.add_argument('--route-a', type=float, default=1.0)
parser.add_argument('--route-b', type=float, default=0.5)
parser.add_argument('--route-T', type=float, default=1.0)
parser.add_argument('--route-margin', type=float, default=0.0)
parser.add_argument('--resume', default='auto',
                    help='checkpoint path or "auto" to resume from model-dir/checkpoint-last.pt')
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
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


# =========================================================
# Main
# =========================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_dir, exist_ok=True)

    def build_lut(classes, device_):
        lut = torch.full((10,), -1, device=device_, dtype=torch.long)
        for i, c in enumerate(classes):
            lut[c] = i
        return lut

    lut4 = build_lut(VEHICLE_CLASSES, device)
    lut6 = build_lut(ANIMAL_CLASSES, device)

    # Stage 1: weaker augmentation for 4/6-class submodels
    transform_sub = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.05, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
    ])

    # Stage 2: stronger augmentation for fusion/TRADES
    transform_fusion = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    base_train_sub = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_sub
    )
    base_train_fusion = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_fusion
    )
    test_set = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test
    )

    train_loader_4 = torch.utils.data.DataLoader(
        CIFARUnknownDataset(base_train_sub, VEHICLE_CLASSES),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    train_loader_6 = torch.utils.data.DataLoader(
        CIFARUnknownDataset(base_train_sub, ANIMAL_CLASSES),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    train_loader_10 = torch.utils.data.DataLoader(
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
    m4 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=5).to(device)
    m6 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=7).to(device)

    if resume_loaded and checkpoint is not None:
        m4.load_state_dict(checkpoint['m4_state_dict'])
        m6.load_state_dict(checkpoint['m6_state_dict'])

    if not resume_loaded:
        opt4 = optim.SGD(m4.parameters(), lr=args.lr,
                        momentum=args.momentum, weight_decay=args.weight_decay)
        opt6 = optim.SGD(m6.parameters(), lr=args.lr,
                        momentum=args.momentum, weight_decay=args.weight_decay)

        print('==== Stage 1 ====')
        for ep in range(1, args.epochs_sub + 1):
            l4, a4 = train_ce_epoch(m4, train_loader_4, opt4, device)
            l6, a6 = train_ce_epoch(m6, train_loader_6, opt6, device)
            print(f'[Sub][{ep}] 4c+unk acc={a4*100:.2f}% | 6c+unk acc={a6*100:.2f}%')

        torch.save(m4.state_dict(), f'{args.model_dir}/wrn4_final.pt')
        torch.save(m6.state_dict(), f'{args.model_dir}/wrn6_final.pt')

    # ================= Stage 2 =================
    fusion = SoftRoutingFusion(m4, m6, a=args.route_a, b=args.route_b, T=args.route_T, margin=args.route_margin).to(device)

    for p in fusion.m4.parameters():
        p.requires_grad = True
    for p in fusion.m6.parameters():
        p.requires_grad = True

    params = [
        {'params': fusion.m4.parameters(), 'lr': args.lr},
        {'params': fusion.m6.parameters(), 'lr': args.lr},
    ]

    optimizer = optim.SGD(
        params,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    base_group_lrs = [pg['lr'] for pg in optimizer.param_groups]

    ema = EMA(fusion, decay=0.999)

    if resume_loaded and checkpoint is not None:
        fusion.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            ckpt_opt = checkpoint['optimizer_state_dict']
            n_cur, n_ckpt = len(optimizer.param_groups), len(ckpt_opt.get('param_groups', []))
            if n_cur != n_ckpt:
                raise ValueError(f'[RESUME] optimizer param_groups mismatch: current={n_cur}, checkpoint={n_ckpt}')
            optimizer.load_state_dict(ckpt_opt)
            base_group_lrs = [pg['lr'] for pg in optimizer.param_groups]
            print(f'[RESUME] optimizer loaded: {n_cur} param_groups, lr={base_group_lrs}')
        if 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']

    # -------- CE warmup (10 epochs, BN NOT frozen) --------
    print('==== CE warmup (10 epochs) ====')
    for ep in range(warmup_start, 11):
        fusion.train()
        for x, y in train_loader_10:
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
            'm4_state_dict': m4.state_dict(),
            'm6_state_dict': m6.state_dict(),
            'model_state_dict': fusion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_shadow': ema.shadow,
            'stage2_epoch': ep,
        }
        torch.save(ckpt, os.path.join(args.model_dir, 'checkpoint-last.pt'))

    # -------- Freeze BN AFTER warmup --------
    freeze_bn(fusion)

    # -------- TRADES training with MultiStepLR (like previous code) --------
    milestones = [args.epochs_fusion // 2, int(args.epochs_fusion * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    if resume_loaded and checkpoint is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        expected_last = start_epoch - 1
        actual_last = getattr(scheduler, 'last_epoch', getattr(scheduler, '_last_epoch', None))
        print(f'[RESUME] scheduler loaded: last_epoch={actual_last}, expected={expected_last} (resume from TRADES ep {start_epoch})')
        if actual_last is not None and actual_last != expected_last:
            print(f'[RESUME] WARNING: scheduler last_epoch mismatch, LR curve may be offset!')

    print('==== TRADES training ====')
    bn_unfrozen = start_epoch > 40
    if bn_unfrozen:
        unfreeze_bn(fusion)
        print(f'[INFO] Resumed past epoch 40, BN already unfrozen')
    for ep in range(start_epoch, args.epochs_fusion + 1):
        fusion.train()
        if not bn_unfrozen:
            freeze_bn(fusion)
        total, correct = 0, 0

        ratio = backbone_lr_ratio(ep, args.epochs_fusion, r1=0.15, r2=0.5, r3=0.35)
        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = base_group_lrs[i] * ratio

        for x, y in train_loader_10:
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
                data_mean=CIFAR10_MEAN,
                data_std=CIFAR10_STD
            )

            out4, out6, _ = fusion(x, return_aux=True)
            loss += aux_ce_loss(out4, out6, y, lut4, lut6)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)

        if not bn_unfrozen and ep >= 40:
            unfreeze_bn(fusion)
            bn_unfrozen = True
            print(f'[INFO] Unfroze BN at epoch {ep}')

        base_lr_before = base_group_lrs[0]
        if scheduler is not None:
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = base_group_lrs[i]
            scheduler.step()
            base_group_lrs = [pg['lr'] for pg in optimizer.param_groups]

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

        current_lr = base_lr_before * ratio
        acc = test_correct / test_total
        print(f'[Fusion/TRADES][{ep}/{args.epochs_fusion}] acc={acc*100:.2f}%, lr={current_lr:.6f}')

        ckpt = {
            'm4_state_dict': m4.state_dict(),
            'm6_state_dict': m6.state_dict(),
            'model_state_dict': fusion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
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
            torch.save(fusion.state_dict(),
                       f'{args.model_dir}/fusion-epoch{ep}.pt')


if __name__ == '__main__':
    main()
