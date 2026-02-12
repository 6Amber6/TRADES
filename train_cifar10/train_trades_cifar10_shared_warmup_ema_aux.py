from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torchvision import transforms

from models.shared_wrn_gated import SharedWRNGated
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
            # Freeze running stats only; keep affine params trainable
            m.eval()

def unfreeze_bn(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.train()


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


def gate_supervision_loss(gate, y, device):
    target = torch.isin(y, torch.tensor(VEHICLE_CLASSES, device=device)).float().view(-1, 1)
    return F.binary_cross_entropy(gate, target)


def gate_kl_loss(gate_adv, gate_clean, eps=1e-6):
    p = gate_adv.clamp(eps, 1 - eps)
    q = gate_clean.clamp(eps, 1 - eps)
    kl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    return kl.mean()


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
def train_ce_epoch(model, loader, optimizer, device, head):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, head=head)
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
    fusion = SharedWRNGated(depth=34, widen_factor=10).to(device)

    stage1_params = fusion.shared_parameters() + fusion.vehicle_parameters() + fusion.animal_parameters()
    opt_stage1 = optim.SGD(stage1_params, lr=args.lr,
                           momentum=args.momentum, weight_decay=args.weight_decay)

    print('==== Stage 1 ====')
    for ep in range(1, args.epochs_sub + 1):
        fusion.train()
        total4, correct4, loss4_sum = 0, 0, 0.0
        total6, correct6, loss6_sum = 0, 0, 0.0

        it4 = iter(train_loader_4)
        it6 = iter(train_loader_6)
        steps = max(len(train_loader_4), len(train_loader_6))

        for _ in range(steps):
            try:
                x4, y4 = next(it4)
            except StopIteration:
                it4 = iter(train_loader_4)
                x4, y4 = next(it4)

            try:
                x6, y6 = next(it6)
            except StopIteration:
                it6 = iter(train_loader_6)
                x6, y6 = next(it6)

            x4, y4 = x4.to(device), y4.to(device)
            x6, y6 = x6.to(device), y6.to(device)

            opt_stage1.zero_grad()

            logits4 = fusion(x4, head='vehicle')
            logits6 = fusion(x6, head='animal')

            loss4 = F.cross_entropy(logits4, y4)
            loss6 = F.cross_entropy(logits6, y6)
            loss = loss4 + loss6
            loss.backward()
            opt_stage1.step()

            loss4_sum += loss4.item() * x4.size(0)
            loss6_sum += loss6.item() * x6.size(0)
            total4 += x4.size(0)
            total6 += x6.size(0)
            correct4 += (logits4.argmax(1) == y4).sum().item()
            correct6 += (logits6.argmax(1) == y6).sum().item()

        a4 = correct4 / total4 if total4 else 0.0
        a6 = correct6 / total6 if total6 else 0.0
        print(f'[Sub][{ep}] 4c acc={a4*100:.2f}% | 6c acc={a6*100:.2f}%')

    torch.save(fusion.state_dict(), f'{args.model_dir}/shared_stage1_final.pt')

    # ================= Stage 2 =================
    # Use different learning rates: heads+gate get full lr, shared backbone gets lr*0.2
    params = [
        {'params': fusion.vehicle_parameters(), 'lr': args.lr},
        {'params': fusion.animal_parameters(), 'lr': args.lr},
        {'params': fusion.gate_parameters(), 'lr': args.lr},
        {'params': fusion.shared_parameters(), 'lr': args.lr * 0.2},
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
    gate_sup_weight = 0.1
    gate_stab_weight = 0.1
    for ep in range(1, args.epochs_fusion + 1):
        fusion.train()
        if not bn_unfrozen:
            freeze_bn(fusion)
        total, correct = 0, 0

        for x, y in train_loader_10:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss, x_adv = trades_loss(
                model=fusion,
                x_natural=x,
                y=y,
                optimizer=optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta,
                data_mean=CIFAR10_MEAN,
                data_std=CIFAR10_STD,
                return_x_adv=True
            )

            out4_adv, out6_adv, _, gate_adv = fusion(x_adv, return_aux=True, return_gate=True)
            loss += aux_ce_loss(out4_adv, out6_adv, y, device)

            _, _, _, gate_clean = fusion(x, return_aux=True, return_gate=True)
            loss += gate_sup_weight * gate_supervision_loss(gate_clean, y, device)
            loss += gate_stab_weight * gate_kl_loss(gate_adv, gate_clean.detach())

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)

            with torch.no_grad():
                # Evaluate using EMA weights for more stable accuracy tracking
                ema.apply_to(fusion)
                pred = fusion(x, return_aux=True)[-1].argmax(1)
                ema.restore(fusion)
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

        # Save raw weights every epoch
        torch.save(fusion.state_dict(),
                   f'{args.model_dir}/fusion-epoch{ep}.pt')

        # Also save EMA weights every epoch
        with torch.no_grad():
            ema.apply_to(fusion)
            torch.save(fusion.state_dict(),
                       f'{args.model_dir}/fusion-ema-epoch{ep}.pt')
            ema.restore(fusion)


if __name__ == '__main__':
    main()
