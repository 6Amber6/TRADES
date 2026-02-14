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
# Dataset wrapper
# =========================================================
class CIFARSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, keep_classes):
        self.base = base_dataset
        self.map = {c: i for i, c in enumerate(keep_classes)}

        targets = torch.tensor(self.base.targets)
        self.idx = (
            (targets.unsqueeze(1) == torch.tensor(keep_classes))
            .any(dim=1)
            .nonzero(as_tuple=False)
            .view(-1)
        )

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, y = self.base[self.idx[i]]
        return x, self.map[y]


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
# Aux loss
# =========================================================
def aux_ce_loss(logits4, logits6, y, device, weight=0.02):
    loss = 0.0

    if logits4 is not None:
        mask4 = torch.isin(y, torch.tensor(VEHICLE_CLASSES, device=device))
        if mask4.any():
            remap4 = torch.tensor(VEHICLE_CLASSES, device=device)
            loss += F.cross_entropy(
                logits4[mask4],
                torch.searchsorted(remap4, y[mask4])
            )

    if logits6 is not None:
        mask6 = torch.isin(y, torch.tensor(ANIMAL_CLASSES, device=device))
        if mask6.any():
            remap6 = torch.tensor(ANIMAL_CLASSES, device=device)
            loss += F.cross_entropy(
                logits6[mask6],
                torch.searchsorted(remap6, y[mask6])
            )

    return weight * loss


# =========================================================
# Gated fusion model
# =========================================================
class GatedFusionWRN(nn.Module):
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
parser = argparse.ArgumentParser(description='TRADES Parallel WRN Training (gated fusion)')
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
parser.add_argument('--model-dir', default='./model-parallel-gated')
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
        logits = model(x)
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
        CIFARSubset(base_train_sub, VEHICLE_CLASSES),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    train_loader_6 = torch.utils.data.DataLoader(
        CIFARSubset(base_train_sub, ANIMAL_CLASSES),
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

    # ================= Stage 1 =================
    m4 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=4).to(device)
    m6 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=6).to(device)

    resume_path = args.resume
    if resume_path == 'auto':
        resume_path = os.path.join(args.model_dir, 'checkpoint-last.pt')

    resume_loaded = False
    start_epoch = 1
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(f'[INFO] Resuming from {resume_path}')
            m4.load_state_dict(checkpoint['m4_state_dict'])
            m6.load_state_dict(checkpoint['m6_state_dict'])
            resume_loaded = True
            start_epoch = checkpoint.get('epoch', 0) + 1

    if not resume_loaded:
        opt4 = optim.SGD(m4.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)
        opt6 = optim.SGD(m6.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)

        print('==== Stage 1 ====')
        for ep in range(1, args.epochs_sub + 1):
            l4, a4 = train_ce_epoch(m4, train_loader_4, opt4, device)
            l6, a6 = train_ce_epoch(m6, train_loader_6, opt6, device)
            print(f'[Sub][{ep}] 4c acc={a4*100:.2f}% | 6c acc={a6*100:.2f}%')

        torch.save(m4.state_dict(), f'{args.model_dir}/wrn4_final.pt')
        torch.save(m6.state_dict(), f'{args.model_dir}/wrn6_final.pt')

    # ================= Stage 2 =================
    fusion = GatedFusionWRN(m4, m6).to(device)

    for p in fusion.m4.parameters():
        p.requires_grad = True
    for p in fusion.m6.parameters():
        p.requires_grad = True

    params = [
        {'params': fusion.fc.parameters(), 'lr': args.lr},
        {'params': fusion.m4.parameters(), 'lr': args.lr},
        {'params': fusion.m6.parameters(), 'lr': args.lr},
        {'params': fusion.gate.parameters(), 'lr': args.lr},
    ]

    optimizer = optim.SGD(
        params,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    ema = EMA(fusion, decay=0.999)
    if resume_loaded:
        fusion.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ema.shadow = checkpoint.get('ema_shadow', ema.shadow)

    # -------- CE warmup (10 epochs, BN NOT frozen) --------
    if not resume_loaded:
        print('==== CE warmup (10 epochs) ====')
        for ep in range(1, 11):
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

        # -------- Freeze BN AFTER warmup --------
        freeze_bn(fusion)
    else:
        # Ensure BN is in the correct state after resume
        if start_epoch <= 40:
            freeze_bn(fusion)

    # -------- TRADES training with MultiStepLR (like previous code) --------
    milestones = [args.epochs_fusion // 2, int(args.epochs_fusion * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if resume_loaded and isinstance(checkpoint, dict):
        sched_state = checkpoint.get('scheduler_state_dict')
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)

    print('==== TRADES training ====')
    bn_unfrozen = False
    bn_unfrozen = start_epoch >= 41
    for ep in range(start_epoch, args.epochs_fusion + 1):
        fusion.train()
        if not bn_unfrozen:
            freeze_bn(fusion)
        total, correct = 0, 0

        ratio = backbone_lr_ratio(ep, args.epochs_fusion, r1=0.15, r2=0.5, r3=0.35)
        fusion_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[1]['lr'] = fusion_lr * ratio
        optimizer.param_groups[2]['lr'] = fusion_lr * ratio

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
            loss += aux_ce_loss(out4, out6, y, device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)

        # Unfreeze BN after 40 TRADES epochs (like previous code)
        # This allows BN stats to adapt to adversarial training
        if not bn_unfrozen and ep >= 40:
            unfreeze_bn(fusion)
            bn_unfrozen = True
            print(f'[INFO] Unfroze BN at epoch {ep}')
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
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
        print(f'[Fusion/TRADES][{ep}/{args.epochs_fusion}] acc={acc*100:.2f}%, lr={current_lr:.6f}')

        # For the last epoch, apply EMA weights before saving
        if ep == args.epochs_fusion:
            # Apply EMA weights to model
            with torch.no_grad():
                for n, p in fusion.named_parameters():
                    if n in ema.shadow:
                        p.data.copy_(ema.shadow[n])
            print(f'[INFO] Applied EMA weights to final checkpoint')

        torch.save(fusion.state_dict(),
                   f'{args.model_dir}/fusion-epoch{ep}.pt')
        torch.save(
            {
                'epoch': ep,
                'model_state_dict': fusion.state_dict(),
                'm4_state_dict': m4.state_dict(),
                'm6_state_dict': m6.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'ema_shadow': ema.shadow,
            },
            os.path.join(args.model_dir, 'checkpoint-last.pt')
        )


if __name__ == '__main__':
    main()
