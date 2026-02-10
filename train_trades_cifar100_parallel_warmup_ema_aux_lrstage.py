from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms

from models.parallel_wrn import WRNWithEmbedding
from trades import trades_loss


# =========================================================
# CIFAR-100 (coarse) class split: 20 coarse labels -> 2 groups of 10
# =========================================================
COARSE_GROUP_A = [0, 1, 7, 8, 11, 12, 13, 14, 15, 16]
COARSE_GROUP_B = [2, 3, 4, 5, 6, 9, 10, 17, 18, 19]
COARSE_GROUP_A_T = torch.tensor(COARSE_GROUP_A, dtype=torch.long)
COARSE_GROUP_B_T = torch.tensor(COARSE_GROUP_B, dtype=torch.long)


# =========================================================
# CIFAR-100 Normalize
# =========================================================
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


# =========================================================
# CIFAR-100 fine -> coarse mapping (fallback for older torchvision)
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
# Fusion model for 20 coarse classes
# =========================================================
class ParallelFusionWRNCoarse(nn.Module):
    def __init__(self, model_a, model_b, num_classes=20):
        super().__init__()
        self.m4 = model_a
        self.m6 = model_b

        for p in self.m4.parameters():
            p.requires_grad = False
        for p in self.m6.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(640 * 2, num_classes)

    def forward(self, x, return_aux=False):
        e4, out4 = self.m4(x, return_embedding=True)
        e6, out6 = self.m6(x, return_embedding=True)
        emb = torch.cat([e4, e6], dim=1)
        out = self.fc(emb)
        if return_aux:
            return out4, out6, out
        return out


# =========================================================
# Aux loss for two coarse groups
# =========================================================
def aux_ce_loss(logits_a, logits_b, y, group_a_t, group_b_t, map_a, map_b, weight=0.02):
    loss = 0.0

    if logits_a is not None:
        mask_a = torch.isin(y, group_a_t)
        if mask_a.any():
            y_a = map_a[y[mask_a]]
            loss += F.cross_entropy(
                logits_a[mask_a],
                y_a
            )

    if logits_b is not None:
        mask_b = torch.isin(y, group_b_t)
        if mask_b.any():
            y_b = map_b[y[mask_b]]
            loss += F.cross_entropy(
                logits_b[mask_b],
                y_b
            )

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
parser = argparse.ArgumentParser(description='TRADES Parallel WRN CIFAR-100 (coarse) Training')
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
parser.add_argument('--model-dir', default='./model-cifar100-parallel')
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


def build_cifar100_coarse(root, train, transform):
    try:
        return torchvision.datasets.CIFAR100(
            root=root, train=train, download=True, transform=transform, target_type='coarse'
        )
    except TypeError:
        dataset = torchvision.datasets.CIFAR100(
            root=root, train=train, download=True, transform=transform
        )
        dataset.targets = [CIFAR100_FINE_TO_COARSE[t] for t in dataset.targets]
        return dataset


# =========================================================
# Main
# =========================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_dir, exist_ok=True)

    # Stage 1: weaker augmentation for 10-class coarse submodels
    transform_sub = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        transforms.RandomErasing(p=0.05, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
    ])

    # Stage 2: stronger augmentation for fusion/TRADES
    transform_fusion = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
    ])

    base_train_sub = build_cifar100_coarse('../data', train=True, transform=transform_sub)
    base_train_fusion = build_cifar100_coarse('../data', train=True, transform=transform_fusion)
    group_a_t = COARSE_GROUP_A_T.to(device)
    group_b_t = COARSE_GROUP_B_T.to(device)
    map_a = torch.full((20,), -1, device=device, dtype=torch.long)
    map_b = torch.full((20,), -1, device=device, dtype=torch.long)
    map_a[group_a_t] = torch.arange(10, device=device)
    map_b[group_b_t] = torch.arange(10, device=device)

    train_loader_a = torch.utils.data.DataLoader(
        CIFARSubset(base_train_sub, COARSE_GROUP_A),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    train_loader_b = torch.utils.data.DataLoader(
        CIFARSubset(base_train_sub, COARSE_GROUP_B),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    train_loader_20 = torch.utils.data.DataLoader(
        base_train_fusion,
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    # ================= Stage 1 =================
    m_a = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=10).to(device)
    m_b = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=10).to(device)

    opt_a = optim.SGD(m_a.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
    opt_b = optim.SGD(m_b.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

    print('==== Stage 1 (CIFAR-100 coarse) ====')
    for ep in range(1, args.epochs_sub + 1):
        l_a, a_a = train_ce_epoch(m_a, train_loader_a, opt_a, device)
        l_b, a_b = train_ce_epoch(m_b, train_loader_b, opt_b, device)
        print(f'[Sub][{ep}] A(10c) acc={a_a*100:.2f}% | B(10c) acc={a_b*100:.2f}%')

    torch.save(m_a.state_dict(), f'{args.model_dir}/wrnA_final.pt')
    torch.save(m_b.state_dict(), f'{args.model_dir}/wrnB_final.pt')

    # ================= Stage 2 =================
    fusion = ParallelFusionWRNCoarse(m_a, m_b, num_classes=20).to(device)

    for p in fusion.m4.parameters():
        p.requires_grad = True
    for p in fusion.m6.parameters():
        p.requires_grad = True

    params = [
        {'params': fusion.fc.parameters(), 'lr': args.lr},
        {'params': fusion.m4.parameters(), 'lr': args.lr},
        {'params': fusion.m6.parameters(), 'lr': args.lr},
    ]

    optimizer = optim.SGD(
        params,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    ema = EMA(fusion, decay=0.999)

    # -------- CE warmup (10 epochs, BN NOT frozen) --------
    print('==== CE warmup (10 epochs) ====')
    for ep in range(1, 11):
        fusion.train()
        for x, y in train_loader_20:
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

    # -------- TRADES training (base LR fixed) --------
    print('==== TRADES training ====')
    bn_unfrozen = False
    for ep in range(1, args.epochs_fusion + 1):
        fusion.train()
        total, correct = 0, 0
        total_loss = 0.0

        ratio = backbone_lr_ratio(ep, args.epochs_fusion, r1=0.15, r2=0.5, r3=0.35)
        optimizer.param_groups[0]['lr'] = args.lr
        fusion_lr = args.lr
        optimizer.param_groups[1]['lr'] = fusion_lr * ratio
        optimizer.param_groups[2]['lr'] = fusion_lr * ratio

        for x, y in train_loader_20:
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

            out_a, out_b, _ = fusion(x, return_aux=True)
            loss += aux_ce_loss(out_a, out_b, y, group_a_t, group_b_t, map_a, map_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)

            total_loss += loss.item() * x.size(0)
            with torch.no_grad():
                ema.apply_to(fusion)
                pred = fusion(x, return_aux=True)[-1].argmax(1)
                ema.restore(fusion)
                correct += (pred == y).sum().item()
                total += y.size(0)

        if not bn_unfrozen and ep >= 40:
            unfreeze_bn(fusion)
            bn_unfrozen = True
            print(f'[INFO] Unfroze BN at epoch {ep}')

        acc = correct / total
        avg_loss = total_loss / total
        print(f'[Fusion/TRADES][{ep}/{args.epochs_fusion}] train_acc={acc*100:.2f}%, train_loss={avg_loss:.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')

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
