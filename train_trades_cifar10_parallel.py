from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms

from models.parallel_wrn import WRNWithEmbedding, ParallelFusionWRN
from trades import trades_loss


# =========================================================
# CIFAR-10 class split
# =========================================================
VEHICLE_CLASSES = [0, 1, 8, 9]          # airplane, automobile, ship, truck
ANIMAL_CLASSES  = [2, 3, 4, 5, 6, 7]    # bird, cat, deer, dog, frog, horse


# =========================================================
# CIFAR-10 Normalize 
# =========================================================
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


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
# BN freeze util (Stage 2 only)
# =========================================================
def freeze_bn(model: torch.nn.Module):
    """
    Freeze all BatchNorm layers: stop updating running_mean/var and keep affine params fixed.
    This matches common robust-training practice to avoid BN stats being poisoned by adversarial examples.
    """
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


# =========================================================
# Arguments (STRICTLY aligned with TRADES baseline)
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
    """Train one epoch with cross-entropy and return avg loss & acc"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def train_trades_epoch(model, loader, optimizer, device,
                       epsilon, num_steps, step_size, beta):
    """Train one epoch with TRADES and return avg loss & clean acc (on natural x)"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        loss = trades_loss(
            model=model,
            x_natural=x,
            y=y,
            optimizer=optimizer,
            step_size=step_size,
            epsilon=epsilon,
            perturb_steps=num_steps,
            beta=beta
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        # clean accuracy on natural examples
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


# =========================================================
# Main
# =========================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_dir, exist_ok=True)

    # Data augmentation identical to TRADES + Normalize (added)
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    base_train = torchvision.datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader_4 = torch.utils.data.DataLoader(
        CIFARSubset(base_train, VEHICLE_CLASSES),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    train_loader_6 = torch.utils.data.DataLoader(
        CIFARSubset(base_train, ANIMAL_CLASSES),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    train_loader_10 = torch.utils.data.DataLoader(
        base_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # =====================================================
    # Stage 1: Train 4-class & 6-class submodels
    # =====================================================
    m4 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=4).to(device)
    m6 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=6).to(device)

    opt4 = optim.SGD(m4.parameters(), lr=args.lr,
                     momentum=args.momentum,
                     weight_decay=args.weight_decay)
    opt6 = optim.SGD(m6.parameters(), lr=args.lr,
                     momentum=args.momentum,
                     weight_decay=args.weight_decay)

    print('==== Stage 1: Training 4-class and 6-class submodels ====')
    for epoch in range(1, args.epochs_sub + 1):
        loss4, acc4 = train_ce_epoch(m4, train_loader_4, opt4, device)
        loss6, acc6 = train_ce_epoch(m6, train_loader_6, opt6, device)

        print(
            f'[Submodels][Epoch {epoch}/{args.epochs_sub}] '
            f'WRN-4: loss={loss4:.4f}, acc={acc4*100:.2f}% | '
            f'WRN-6: loss={loss6:.4f}, acc={acc6*100:.2f}%'
        )

    torch.save(m4.state_dict(), f'{args.model_dir}/wrn4_final.pt')
    torch.save(m6.state_dict(), f'{args.model_dir}/wrn6_final.pt')

    # =====================================================
    # Stage 2: TRADES training for fusion model (Freeze BN)
    # =====================================================
    fusion = ParallelFusionWRN(m4, m6).to(device)

    # Freeze BN for Stage 2 only
    freeze_bn(fusion)

    optimizer_fusion = optim.SGD(
        filter(lambda p: p.requires_grad, fusion.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    print('==== Stage 2: TRADES training for fusion model (Freeze BN) ====')
    for epoch in range(1, args.epochs_fusion + 1):
        loss, acc = train_trades_epoch(
            fusion,
            train_loader_10,
            optimizer_fusion,
            device,
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            beta=args.beta
        )

        print(
            f'[Fusion/TRADES][Epoch {epoch}/{args.epochs_fusion}] '
            f'loss={loss:.4f}, acc={acc*100:.2f}%'
        )

        torch.save(
            fusion.state_dict(),
            f'{args.model_dir}/fusion-epoch{epoch}.pt'
        )


if __name__ == '__main__':
    main()
