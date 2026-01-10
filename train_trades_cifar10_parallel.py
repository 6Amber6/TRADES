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
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()


# =========================================================
# Main
# =========================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_dir, exist_ok=True)

    # Data augmentation identical to TRADES
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
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
    # Stage 1: Train 4-class & 6-class submodels (CE)
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
        train_ce_epoch(m4, train_loader_4, opt4, device)
        train_ce_epoch(m6, train_loader_6, opt6, device)
        print(f'[Submodels] Epoch {epoch}/{args.epochs_sub}')

    torch.save(m4.state_dict(), f'{args.model_dir}/wrn4_final.pt')
    torch.save(m6.state_dict(), f'{args.model_dir}/wrn6_final.pt')

    # =====================================================
    # Stage 2: TRADES training for fusion model
    # =====================================================
    fusion = ParallelFusionWRN(m4, m6).to(device)

    optimizer_fusion = optim.SGD(
        filter(lambda p: p.requires_grad, fusion.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    print('==== Stage 2: TRADES training for fusion model ====')
    for epoch in range(1, args.epochs_fusion + 1):
        fusion.train()
        for x, y in train_loader_10:
            x, y = x.to(device), y.to(device)
            optimizer_fusion.zero_grad()

            loss = trades_loss(
                model=fusion,
                x_natural=x,
                y=y,
                optimizer=optimizer_fusion,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta
            )

            loss.backward()
            optimizer_fusion.step()

        print(f'[Fusion/TRADES] Epoch {epoch}/{args.epochs_fusion}')
        torch.save(
            fusion.state_dict(),
            f'{args.model_dir}/fusion-epoch{epoch}.pt'
        )


if __name__ == '__main__':
    main()
