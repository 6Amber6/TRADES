"""
CIFAR-100 Gated Fusion with 2 sub-models (3-stage training).
Stage 1: Pre-train two experts on their respective class subsets (CE).
Stage 2: Train router (gate + fc) with experts frozen (CE).
Stage 3: Joint TRADES adversarial training with differential LR.

Group A (Nature/Biology, 10 superclasses, 50 fine classes):
  aquatic mammals, fish, flowers, insects, large carnivores,
  large omnivores/herbivores, medium mammals, non-insect invertebrates,
  reptiles, small mammals

Group B (Man-made/Scenes, 10 superclasses, 50 fine classes):
  food containers, fruit/vegetables, household electrical,
  household furniture, large man-made outdoor, large natural outdoor,
  people, trees, vehicles 1, vehicles 2
"""

from __future__ import print_function
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
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
# CIFAR-100 Normalize
# =========================================================
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


# =========================================================
# CIFAR-100 fine -> coarse mapping (100 fine classes -> 20 coarse)
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

# 20 coarse classes:
# 0 aquatic mammals, 1 fish, 2 flowers, 3 food containers, 4 fruit/veg,
# 5 household electrical, 6 household furniture, 7 insects, 8 large carnivores,
# 9 large man-made outdoor, 10 large natural outdoor, 11 large omnivores/herbivores,
# 12 medium mammals, 13 non-insect invertebrates, 14 people, 15 reptiles,
# 16 small mammals, 17 trees, 18 vehicles 1, 19 vehicles 2

# =========================================================
# 2-group partition (10 superclasses each, 50 fine classes each)
# =========================================================
COARSE_GROUPS = {
    # Group A: Nature/Biology
    "nature": [0, 1, 7, 8, 11, 12, 13, 15, 16, 2],
    # aquatic mammals, fish, insects, large carnivores,
    # large omnivores/herbivores, medium mammals, non-insect invertebrates,
    # reptiles, small mammals, flowers

    # Group B: Man-made/Scenes
    "manmade": [3, 4, 5, 6, 9, 10, 14, 17, 18, 19],
    # food containers, fruit/veg, household electrical, household furniture,
    # large man-made outdoor, large natural outdoor, people, trees,
    # vehicles 1, vehicles 2
}
GROUP_ORDER = ["nature", "manmade"]


def build_fine_classes_for_group(group_coarse):
    return sorted([i for i, c in enumerate(CIFAR100_FINE_TO_COARSE) if c in group_coarse])


# Precompute fine classes per group
GROUP_FINE = {name: build_fine_classes_for_group(coarse) for name, coarse in COARSE_GROUPS.items()}


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
# Gated fusion model for 2 groups -> 100 classes
# =========================================================
class GatedFusionWRN100(nn.Module):
    def __init__(self, m_nature, m_manmade, emb_dim=640, hidden_dim=256):
        super().__init__()
        self.m_nature = m_nature
        self.m_manmade = m_manmade
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Linear(emb_dim, 100)

    def forward(self, x, return_aux=False):
        e_nat, out_nat = self.m_nature(x, return_embedding=True)
        e_man, out_man = self.m_manmade(x, return_embedding=True)
        gate_in = torch.cat([e_nat, e_man], dim=1)
        alpha = torch.sigmoid(self.gate(gate_in))
        emb = alpha * e_nat + (1 - alpha) * e_man
        out = self.fc(emb)
        if return_aux:
            return out_nat, out_man, out
        return out


# =========================================================
# Aux loss for 2 groups
# =========================================================
def aux_ce_loss(logits_nature, logits_manmade, y, device, weight=0.02):
    fine_to_coarse = torch.tensor(CIFAR100_FINE_TO_COARSE, device=device)
    coarse_y = fine_to_coarse[y]
    loss = 0.0

    # Nature group
    nature_coarse_t = torch.tensor(COARSE_GROUPS["nature"], device=device)
    mask_nat = torch.isin(coarse_y, nature_coarse_t)
    if mask_nat.any() and logits_nature is not None:
        fine_classes = GROUP_FINE["nature"]
        map_t = torch.full((100,), -1, device=device, dtype=torch.long)
        map_t[torch.tensor(fine_classes, device=device)] = torch.arange(len(fine_classes), device=device)
        loss += F.cross_entropy(logits_nature[mask_nat], map_t[y[mask_nat]])

    # Manmade group
    manmade_coarse_t = torch.tensor(COARSE_GROUPS["manmade"], device=device)
    mask_man = torch.isin(coarse_y, manmade_coarse_t)
    if mask_man.any() and logits_manmade is not None:
        fine_classes = GROUP_FINE["manmade"]
        map_t = torch.full((100,), -1, device=device, dtype=torch.long)
        map_t[torch.tensor(fine_classes, device=device)] = torch.arange(len(fine_classes), device=device)
        loss += F.cross_entropy(logits_manmade[mask_man], map_t[y[mask_man]])

    return weight * loss


# =========================================================
# Backbone LR ratio schedule (Stage 3)
# =========================================================
def backbone_lr_ratio(epoch, total_epochs, r1=0.2, r2=0.2, r3=0.2):
    return r1


# =========================================================
# Test accuracy evaluation
# =========================================================
def eval_test(model, test_loader, device, use_ema=False, ema=None):
    model.eval()
    if use_ema and ema is not None:
        ema.apply_to(model)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    if use_ema and ema is not None:
        ema.restore(model)
    model.train()
    return correct / total


# =========================================================
# Arguments
# =========================================================
parser = argparse.ArgumentParser(description='TRADES CIFAR-100 Gated Fusion 3-Stage (2 groups)')
parser.add_argument('--epochs-sub', type=int, default=100)
parser.add_argument('--epochs-router', type=int, default=25,
                    help='Stage 2: router-only CE training epochs (experts frozen)')
parser.add_argument('--epochs-fusion', type=int, default=100,
                    help='Stage 3: joint TRADES training epochs')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--router-lr', type=float, default=0.1,
                    help='Stage 2 learning rate for router (gate + fc)')
parser.add_argument('--fusion-lr', type=float, default=None,
                    help='Stage 3 base lr (default: same as --lr)')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=10)
parser.add_argument('--step-size', type=float, default=0.007)
parser.add_argument('--beta', type=float, default=6.0)
parser.add_argument('--aux-weight', type=float, default=0.02)
parser.add_argument('--model-dir', default='./model-cifar100-gated-3stage-2group')
parser.add_argument('--sub-depth', type=int, default=22, help='WRN depth')
parser.add_argument('--sub-widen', type=int, default=8, help='WRN widen factor')
parser.add_argument('--arch', default=None, choices=['wrn34-10', 'wrn22-8', 'wrn28-8'],
                    help='Architecture preset')
parser.add_argument('--resume', default='auto',
                    help='checkpoint path or "auto" to resume from model-dir/checkpoint-last.pt')
args = parser.parse_args()

# Resolve arch preset
ARCH_PRESETS = {'wrn34-10': (34, 10), 'wrn22-8': (22, 8), 'wrn28-8': (28, 8)}
if args.arch:
    args.sub_depth, args.sub_widen = ARCH_PRESETS[args.arch]

fusion_lr = args.fusion_lr if args.fusion_lr is not None else args.lr


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
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_dir, exist_ok=True)

    # Precompute fine classes
    fine_nature = GROUP_FINE["nature"]
    fine_manmade = GROUP_FINE["manmade"]
    print(f'Group nature: {len(fine_nature)} fine classes from coarse {COARSE_GROUPS["nature"]}')
    print(f'Group manmade: {len(fine_manmade)} fine classes from coarse {COARSE_GROUPS["manmade"]}')

    # Augmentations
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

    train_loader_nature = torch.utils.data.DataLoader(
        CIFARFineSubset(base_train_sub, fine_nature),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    train_loader_manmade = torch.utils.data.DataLoader(
        CIFARFineSubset(base_train_sub, fine_manmade),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    train_loader_100 = torch.utils.data.DataLoader(
        base_train_fusion,
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Sub-task test loaders for Stage 1 test accuracy
    test_set_sub = torchvision.datasets.CIFAR100(
        root='../data', train=False, download=True, transform=transform_test
    )
    test_loader_nature = torch.utils.data.DataLoader(
        CIFARFineSubset(test_set_sub, fine_nature),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader_manmade = torch.utils.data.DataLoader(
        CIFARFineSubset(test_set_sub, fine_manmade),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # ================= Resume =================
    resume_path = args.resume
    if resume_path == 'auto':
        resume_path = os.path.join(args.model_dir, 'checkpoint-last.pt')

    resume_loaded = False
    checkpoint = None
    resume_stage = None
    resume_ep = 1
    if resume_path and os.path.isfile(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=device)
        except Exception as e:
            print(f'[WARN] Failed to load checkpoint (corrupted?): {e}')
            checkpoint = None
        if checkpoint is not None and isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(f'[INFO] Resuming from {resume_path}')
            resume_loaded = True
            resume_stage = checkpoint.get('current_stage', None)
            resume_ep = checkpoint.get('current_epoch', 1)

    # ================= Stage 1: Expert Pre-training =================
    n_nature = len(fine_nature)
    n_manmade = len(fine_manmade)
    m_nature = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=n_nature).to(device)
    m_manmade = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=n_manmade).to(device)

    if resume_loaded and checkpoint is not None:
        m_nature.load_state_dict(checkpoint['m_nature_state_dict'])
        m_manmade.load_state_dict(checkpoint['m_manmade_state_dict'])

    if not resume_loaded:
        print('==== Stage 1: Expert Pre-training ====')

        opt_nat = optim.SGD(m_nature.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
        opt_man = optim.SGD(m_manmade.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)

        # LR decay for Stage 1: milestones at 50 and 65
        sub_milestones = [int(args.epochs_sub * 0.625), int(args.epochs_sub * 0.8125)]
        sched_nat = MultiStepLR(opt_nat, milestones=sub_milestones, gamma=0.1)
        sched_man = MultiStepLR(opt_man, milestones=sub_milestones, gamma=0.1)
        print(f'[Stage1] LR milestones: {sub_milestones}')

        for ep in range(1, args.epochs_sub + 1):
            l_nat, a_nat = train_ce_epoch(m_nature, train_loader_nature, opt_nat, device)
            l_man, a_man = train_ce_epoch(m_manmade, train_loader_manmade, opt_man, device)

            # Test accuracy on sub-task test sets
            m_nature.eval()
            m_manmade.eval()
            t_nat_correct, t_nat_total = 0, 0
            t_man_correct, t_man_total = 0, 0
            with torch.no_grad():
                for x, y in test_loader_nature:
                    x, y = x.to(device), y.to(device)
                    t_nat_correct += (m_nature(x).argmax(1) == y).sum().item()
                    t_nat_total += y.size(0)
                for x, y in test_loader_manmade:
                    x, y = x.to(device), y.to(device)
                    t_man_correct += (m_manmade(x).argmax(1) == y).sum().item()
                    t_man_total += y.size(0)
            ta_nat = t_nat_correct / t_nat_total
            ta_man = t_man_correct / t_man_total

            print(f'[Stage1][{ep}/{args.epochs_sub}] nature train={a_nat*100:.2f}% test={ta_nat*100:.2f}% | manmade train={a_man*100:.2f}% test={ta_man*100:.2f}% | lr={opt_nat.param_groups[0]["lr"]:.4f}')

            sched_nat.step()
            sched_man.step()

        torch.save(m_nature.state_dict(), f'{args.model_dir}/wrn_nature_final.pt')
        torch.save(m_manmade.state_dict(), f'{args.model_dir}/wrn_manmade_final.pt')

    # ================= Build fusion model =================
    emb_dim = 64 * args.sub_widen
    fusion = GatedFusionWRN100(m_nature, m_manmade, emb_dim=emb_dim).to(device)

    if resume_loaded and checkpoint is not None:
        fusion.load_state_dict(checkpoint['model_state_dict'])

    # ================= Stage 2: Router Training (experts frozen) =================
    skip_stage2 = resume_loaded and resume_stage == 'stage3'
    stage2_start = resume_ep if (resume_loaded and resume_stage == 'stage2') else 1

    if not skip_stage2:
        print(f'==== Stage 2: Router Training ({args.epochs_router} epochs, experts frozen) ====')

        # Freeze expert parameters
        for p in fusion.m_nature.parameters():
            p.requires_grad = False
        for p in fusion.m_manmade.parameters():
            p.requires_grad = False

        # Only train gate + fc
        router_params = [
            {'params': fusion.gate.parameters(), 'lr': args.router_lr},
            {'params': fusion.fc.parameters(), 'lr': args.router_lr},
        ]
        router_optimizer = optim.SGD(
            router_params,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )

        if resume_loaded and resume_stage == 'stage2' and 'optimizer_state_dict' in checkpoint:
            router_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        ema = EMA(fusion, decay=0.999)
        if resume_loaded and resume_stage == 'stage2' and 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']

        for ep in range(stage2_start, args.epochs_router + 1):
            fusion.train()
            fusion.m_nature.eval()
            fusion.m_manmade.eval()

            for x, y in train_loader_100:
                x, y = x.to(device), y.to(device)
                router_optimizer.zero_grad()

                out = fusion(x)
                loss = F.cross_entropy(out, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
                router_optimizer.step()
                ema.update(fusion)

            acc = eval_test(fusion, test_loader, device, use_ema=True, ema=ema)
            print(f'[Stage2][{ep}/{args.epochs_router}] test_acc={acc*100:.2f}%, lr={router_optimizer.param_groups[0]["lr"]:.6f}')

            ckpt = {
                'm_nature_state_dict': m_nature.state_dict(),
                'm_manmade_state_dict': m_manmade.state_dict(),
                'model_state_dict': fusion.state_dict(),
                'optimizer_state_dict': router_optimizer.state_dict(),
                'ema_shadow': ema.shadow,
                'current_stage': 'stage2',
                'current_epoch': ep,
            }
            torch.save(ckpt, os.path.join(args.model_dir, 'checkpoint-last.pt'))

        torch.save(fusion.state_dict(), f'{args.model_dir}/fusion-stage2-final.pt')
        print(f'[INFO] Stage 2 complete. Saved fusion-stage2-final.pt')

    # ================= Stage 3: Joint TRADES Training =================
    print(f'==== Stage 3: Joint TRADES Training ({args.epochs_fusion} epochs) ====')

    # Unfreeze experts
    for p in fusion.m_nature.parameters():
        p.requires_grad = True
    for p in fusion.m_manmade.parameters():
        p.requires_grad = True

    # Differential LR: gate+fc get full lr, experts get lr*ratio
    params = [
        {'params': fusion.fc.parameters(), 'lr': fusion_lr},
        {'params': fusion.m_nature.parameters(), 'lr': fusion_lr},
        {'params': fusion.m_manmade.parameters(), 'lr': fusion_lr},
        {'params': fusion.gate.parameters(), 'lr': fusion_lr},
    ]

    optimizer = optim.SGD(
        params,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    ema = EMA(fusion, decay=0.999)

    stage3_start = 1
    if resume_loaded and resume_stage == 'stage3':
        stage3_start = resume_ep
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']

    # BN stays in training mode throughout Stage 3 (no freeze)
    # Freezing BN causes collapse when experts were trained with LR decay

    milestones = [args.epochs_fusion // 2, int(args.epochs_fusion * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    if resume_loaded and resume_stage == 'stage3' and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if stage3_start > 1:
        for _ in range(1, stage3_start):
            scheduler.step()

    for ep in range(stage3_start, args.epochs_fusion + 1):
        fusion.train()

        ratio = backbone_lr_ratio(ep, args.epochs_fusion)
        cur_fc_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[1]['lr'] = cur_fc_lr * ratio  # m_nature
        optimizer.param_groups[2]['lr'] = cur_fc_lr * ratio  # m_manmade
        optimizer.param_groups[3]['lr'] = cur_fc_lr          # gate

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

            out_nat, out_man, _ = fusion(x, return_aux=True)
            loss += aux_ce_loss(out_nat, out_man, y, device, weight=args.aux_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        acc = eval_test(fusion, test_loader, device, use_ema=True, ema=ema)
        print(f'[Stage3/TRADES][{ep}/{args.epochs_fusion}] test_acc={acc*100:.2f}%, lr={current_lr:.6f}, backbone_lr={cur_fc_lr * ratio:.6f}')

        ckpt = {
            'm_nature_state_dict': m_nature.state_dict(),
            'm_manmade_state_dict': m_manmade.state_dict(),
            'model_state_dict': fusion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_shadow': ema.shadow,
            'current_stage': 'stage3',
            'current_epoch': ep,
        }
        torch.save(ckpt, os.path.join(args.model_dir, 'checkpoint-last.pt'))

        if ep >= 60 and (ep % 5 == 0 or ep == args.epochs_fusion):
            if ep == args.epochs_fusion:
                with torch.no_grad():
                    for n, p in fusion.named_parameters():
                        if n in ema.shadow:
                            p.data.copy_(ema.shadow[n])
                print(f'[INFO] Applied EMA weights to final checkpoint')
            torch.save(
                {
                    "state_dict": fusion.state_dict(),
                    "group_order": GROUP_ORDER,
                    "coarse_groups": COARSE_GROUPS,
                    "args": vars(args),
                },
                f'{args.model_dir}/fusion-epoch{ep}.pt'
            )

    print('Training complete.')


if __name__ == '__main__':
    main()
