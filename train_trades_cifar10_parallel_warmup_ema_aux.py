from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torchvision import transforms

from models.parallel_wrn import WRNWithEmbedding, ParallelFusionWRN
from trades import trades_loss


# =========================================================
# CIFAR-10 class split
# =========================================================
VEHICLE_CLASSES = [0, 1, 8, 9]
ANIMAL_CLASSES  = [2, 3, 4, 5, 6, 7]


# =========================================================
# CIFAR-10 Normalize
# =========================================================
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


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

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)


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
# Arguments
# =========================================================
parser = argparse.ArgumentParser(description='TRADES Parallel WRN Training')
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

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        # Add RandomErasing for better regularization (like previous code)
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    ])

    base_train = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform
    )

    train_loader_4 = torch.utils.data.DataLoader(
        CIFARSubset(base_train, VEHICLE_CLASSES),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    train_loader_6 = torch.utils.data.DataLoader(
        CIFARSubset(base_train, ANIMAL_CLASSES),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    train_loader_10 = torch.utils.data.DataLoader(
        base_train,
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    # ================= Stage 1 =================
    m4 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=4).to(device)
    m6 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=6).to(device)

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
    fusion = ParallelFusionWRN(m4, m6).to(device)
    
    # Enable training for submodels with smaller learning rate (like previous code)
    # This is crucial for better accuracy - backbone should adapt slightly
    for p in fusion.m4.parameters():
        p.requires_grad = True
    for p in fusion.m6.parameters():
        p.requires_grad = True
    
    # Use different learning rates: fusion layer gets full lr, backbones get lr*0.2
    params = [
        {'params': fusion.fc.parameters(), 'lr': args.lr},
        {'params': fusion.m4.parameters(), 'lr': args.lr * 0.2},
        {'params': fusion.m6.parameters(), 'lr': args.lr * 0.2},
    ]
    
    optimizer = optim.SGD(
        params,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True  # Use Nesterov momentum like previous code
    )

    ema = EMA(fusion, decay=0.999)

    # -------- CE warmup (10 epochs, BN NOT frozen) --------
    print('==== CE warmup (10 epochs) ====')
    # Keep constant LR during warmup
    for ep in range(1, 11):
        fusion.train()
        for x, y in train_loader_10:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = fusion(x)
            loss = F.cross_entropy(out, y)

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)

        print(f'[Warmup] Epoch {ep}/10, lr={optimizer.param_groups[0]["lr"]:.6f}')

    # -------- Freeze BN AFTER warmup --------
    freeze_bn(fusion)

    # -------- TRADES training with MultiStepLR (like previous code) --------
    # MultiStepLR is more stable than CosineAnnealingLR for this scenario
    # Decay at 50% and 75% of training
    milestones = [args.epochs_fusion // 2, int(args.epochs_fusion * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    print('==== TRADES training ====')
    bn_unfrozen = False
    for ep in range(1, args.epochs_fusion + 1):
        fusion.train()
        total, correct = 0, 0

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
                beta=args.beta
            )

            out4, out6, _ = fusion(x, return_aux=True)
            loss += aux_ce_loss(out4, out6, y, device)

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)

            with torch.no_grad():
                pred = fusion(x, return_aux=True)[-1].argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        # Unfreeze BN after 40 TRADES epochs (like previous code)
        # This allows BN stats to adapt to adversarial training
        if not bn_unfrozen and ep >= 40:
            unfreeze_bn(fusion)
            bn_unfrozen = True
            print(f'[INFO] Unfroze BN at epoch {ep}')
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        acc = correct / total
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


if __name__ == '__main__':
    main()
