# CIFAR-100: 4 experts x 25 classes, each with unknown, routing by score_m1 = 1 - unknown_m
# Variant: uniform known class sampling (500 per class) + aux_weight=0.10, unknown_weight=0.3 + 26-class acc every 10 ep

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
# 4 groups: 25 fine classes each, + unknown
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
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
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
# Dataset: known (25) + unknown for each group
# =========================================================
class CIFARUnknownSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, keep_fine_classes):
        self.base = base_dataset
        self.known = list(keep_fine_classes)
        self.map = {c: i for i, c in enumerate(self.known)}
        self.known_set = set(self.known)
        self.unknown_idx = len(self.known)
        self.known_indices = []
        self.unknown_indices = []
        targets = getattr(base_dataset, 'targets', None)
        if targets is None:
            for i in range(len(base_dataset)):
                _, y = base_dataset[i]
                if int(y) in self.known_set:
                    self.known_indices.append(i)
                else:
                    self.unknown_indices.append(i)
        else:
            for i, y in enumerate(targets):
                if int(y) in self.known_set:
                    self.known_indices.append(i)
                else:
                    self.unknown_indices.append(i)

    def get_known_by_class(self):
        """Partition known_indices by local class id (0..24). For uniform sampling."""
        known_by_class = {i: [] for i in range(len(self.known))}
        targets = getattr(self.base, 'targets', None)
        if targets is None:
            for idx in self.known_indices:
                _, y = self.base[idx]
                local = self.map[int(y)]
                known_by_class[local].append(idx)
        else:
            for idx in self.known_indices:
                local = self.map[int(targets[idx])]
                known_by_class[local].append(idx)
        return known_by_class

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        y = int(y)
        if y in self.known_set:
            y = self.map[y]
        else:
            y = self.unknown_idx
        return x, y


# =========================================================
# Uniform known class sampler: each known class 500 samples per epoch (uniform),
# 2:1 known vs unknown per batch (known:unknown = 2:1).
# =========================================================
class UniformKnownClassSampler(torch.utils.data.Sampler):
    def __init__(self, subset, batch_size, shuffle=True):
        self.known_by_class = subset.get_known_by_class()
        self.unknown_indices = list(subset.unknown_indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_known_classes = len(self.known_by_class)
        self.n_unknown_per_batch = batch_size // 3
        self.n_known_per_batch = batch_size - self.n_unknown_per_batch
        self.samples_per_known_class = 500
        self.n_known_total = self.n_known_classes * self.samples_per_known_class
        self.n_unknown_total = self.n_known_total // 2  # 2:1 ratio
        self.n_batches = self.n_unknown_total // self.n_unknown_per_batch
        self.n_batches = max(1, self.n_batches)

    def __iter__(self):
        import random
        n_known_per_batch = self.n_known_per_batch
        n_unknown_per_batch = self.n_unknown_per_batch
        known_by_class = {k: list(v) for k, v in self.known_by_class.items()}
        if self.shuffle:
            for k in known_by_class:
                random.shuffle(known_by_class[k])
        unknown = self.unknown_indices.copy()
        if self.shuffle:
            random.shuffle(unknown)
        n_unknown = len(unknown)

        n_high = n_known_per_batch % self.n_known_classes
        if n_high == 0:
            n_high = self.n_known_classes
        n_low = self.n_known_classes - n_high
        per_high = (n_known_per_batch + self.n_known_classes - 1) // self.n_known_classes
        per_low = per_high - 1 if n_low > 0 else per_high

        class_order = list(range(self.n_known_classes))
        if self.shuffle:
            random.shuffle(class_order)

        for b in range(self.n_batches):
            batch = []
            for c_idx, c in enumerate(class_order):
                n_take = per_high if c_idx < n_high else per_low
                arr = known_by_class[c]
                for i in range(n_take):
                    pos = (b * n_take + i) % len(arr)
                    batch.append(arr[pos])
            for i in range(n_unknown_per_batch):
                batch.append(unknown[(b * n_unknown_per_batch + i) % n_unknown])
            if self.shuffle:
                random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# =========================================================
# BN freeze/unfreeze
# =========================================================
def freeze_bn_stats_only(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            if getattr(m, "affine", True) and m.weight is not None and m.bias is not None:
                m.weight.requires_grad = True
                m.bias.requires_grad = True

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def unfreeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()
            for p in m.parameters():
                p.requires_grad = True


# =========================================================
# Routing M1: score_i = 1 - unknown_i (confidence in "not unknown")
# =========================================================
def routing_scores_m1(unk_probs, T=1.0):
    """
    unk_probs: [B, 4] each expert's unknown probability
    score_i = 1 - unk_i
    """
    scores = 1.0 - unk_probs
    w = F.softmax(scores / T, dim=1)
    return w


# =========================================================
# Fusion model: 4 experts (25+1 each), routing M1
# =========================================================
class UnknownRoutingFusion100(nn.Module):
    def __init__(self, submodels, owned_fine_per_group, T=1.0, margin=0.0):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        self.owned_fine = owned_fine_per_group
        self.T = T
        self.margin = margin

    def set_temperature(self, T):
        self.T = T

    def forward(self, x, return_aux=False):
        group_logits = []
        unk_probs = []
        for m, owned in zip(self.submodels, self.owned_fine):
            logits = m(x)
            p = F.softmax(logits, dim=1)
            unk_probs.append(p[:, -1:])
            group_logits.append(logits)

        unk_probs = torch.cat(unk_probs, dim=1)
        w = routing_scores_m1(unk_probs, T=self.T)

        B = group_logits[0].size(0)
        device = group_logits[0].device
        final = torch.zeros(B, 100, device=device, dtype=group_logits[0].dtype)
        for i, (loc, owned) in enumerate(zip(group_logits, self.owned_fine)):
            known = loc[:, :len(owned)]
            std = known.std(dim=1, keepdim=True).clamp(min=1e-6)
            known = known / std
            final[:, owned] += w[:, i:i+1] * known

        if return_aux:
            return group_logits, final
        return final


# =========================================================
# Aux loss: positive (known) + negative (unknown) supervision
# aux_weight=0.10, unknown_weight=0.3
# =========================================================
def aux_ce_loss_unk(group_logits, y, group_defs, fine_to_coarse_t, weight=0.10, unknown_weight=0.3):
    loss = 0.0
    n_experts = len(group_logits)
    for i, (logits, g) in enumerate(zip(group_logits, group_defs)):
        coarse_y = fine_to_coarse_t[y]
        mask_in = torch.isin(coarse_y, g["coarse_t"])
        mask_out = ~mask_in

        if mask_in.any():
            y_local = g["map_t"][y[mask_in]]
            loss += F.cross_entropy(logits[mask_in], y_local)

        if mask_out.any():
            unk_idx = logits.size(1) - 1
            target_unk = torch.full((mask_out.sum().item(),), unk_idx, device=logits.device, dtype=torch.long)
            loss += unknown_weight * F.cross_entropy(logits[mask_out], target_unk)
    return weight * (loss / n_experts)


# =========================================================
# Arguments
# =========================================================
parser = argparse.ArgumentParser(description='CIFAR-100 4x25+unk routing M1 (uniform, aux_weight=0.10, unk_weight=0.3) TRADES')
parser.add_argument('--epochs-sub', type=int, default=80)
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
parser.add_argument('--aux-weight', type=float, default=0.10)
parser.add_argument('--aux-unknown-weight', type=float, default=0.3,
                    help='down-weight out-of-domain->Unknown loss')
parser.add_argument('--route-T', type=float, default=1.0)
parser.add_argument('--route-T-start', type=float, default=1.0)
parser.add_argument('--route-T-end', type=float, default=0.2)
parser.add_argument('--route-margin', type=float, default=0.0)
parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'none'])
parser.add_argument('--model-dir', default='./model-cifar100-4x25-unk-m1-uniform')
parser.add_argument('--resume', default='auto')
args = parser.parse_args()


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
        total += y.size(0)
    return total_loss / total, correct / total


def build_fine_classes_for_group(group_coarse):
    return [i for i, c in enumerate(CIFAR100_FINE_TO_COARSE) if c in group_coarse]


def eval_26class_acc(model, loader, device, num_classes=26):
    """Compute per-class accuracy for 26 classes (25 known + 1 unknown)."""
    model.eval()
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for c in range(num_classes):
                mask = (y == c)
                if mask.any():
                    total_per_class[c] += mask.sum().item()
                    correct_per_class[c] += (pred[mask] == c).sum().item()
    acc_per_class = [correct_per_class[c] / total_per_class[c] * 100 if total_per_class[c] > 0 else 0.0 for c in range(num_classes)]
    return acc_per_class


def eval_routing_quality(fusion_model, test_group_loaders, test_loader, device, group_names):
    """Evaluate routing quality: global_w (full test set) + domain_w (each expert on its in-domain samples). M1: score=1-unk."""
    fusion_model.eval()

    # 1. global_w: average routing weights on full test set
    all_w = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            unk_probs = []
            for m in fusion_model.submodels:
                p = F.softmax(m(x), dim=1)
                unk_probs.append(p[:, -1:])
            unk_probs = torch.cat(unk_probs, dim=1)
            w = routing_scores_m1(unk_probs, T=fusion_model.T)
            all_w.append(w.cpu())
    global_w = torch.cat(all_w).mean(0)

    # 2. domain_w: each expert's weight on its own in-domain samples (y < 25, exclude unknown)
    domain_ws = []
    for i, loader in enumerate(test_group_loaders):
        dw = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                mask = y < 25  # in-domain only, exclude unknown
                if not mask.any():
                    continue
                x = x[mask]
                unk_probs = []
                for m in fusion_model.submodels:
                    p = F.softmax(m(x), dim=1)
                    unk_probs.append(p[:, -1:])
                unk_probs = torch.cat(unk_probs, dim=1)
                w = routing_scores_m1(unk_probs, T=fusion_model.T)
                dw.append(w[:, i].cpu())
        domain_ws.append(torch.cat(dw).mean().item() if dw else 0.0)

    return global_w, domain_ws


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

    base_train_sub = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_sub)
    base_train_fusion = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_fusion)
    test_set = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)

    group_names = GROUP_ORDER
    group_coarse = [COARSE_GROUPS[name] for name in group_names]
    group_fine = [build_fine_classes_for_group(g) for g in group_coarse]

    group_loaders = []
    for fine_classes in group_fine:
        subset = CIFARUnknownSubset(base_train_sub, fine_classes)
        sampler = UniformKnownClassSampler(subset, batch_size=args.batch_size, shuffle=True)
        loader = torch.utils.data.DataLoader(
            subset,
            batch_sampler=sampler,
            num_workers=2, pin_memory=True
        )
        group_loaders.append(loader)

    train_loader_100 = torch.utils.data.DataLoader(
        base_train_fusion, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_group_loaders = []
    for fine_classes in group_fine:
        test_subset = CIFARUnknownSubset(test_set, fine_classes)
        test_group_loaders.append(torch.utils.data.DataLoader(
            test_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
        ))

    # Resume
    resume_path = args.resume if args.resume != 'auto' else os.path.join(args.model_dir, 'checkpoint-last.pt')
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

    # Stage 1: 4 submodels, each 25+1=26 classes
    submodels = []
    for i in range(4):
        m = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=len(group_fine[i]) + 1).to(device)
        submodels.append(m)

    if not resume_loaded:
        print('==== Stage 1 (4x25+unk pretrain, uniform known 500/class, 2:1 known/unknown) ====')
        for i, loader in enumerate(group_loaders):
            subset = loader.dataset
            kbc = subset.get_known_by_class()
            per_class = [len(kbc[c]) for c in range(len(kbc))]
            print(f'  {group_names[i]}: known={len(subset.known_indices)} (per_class={per_class[:5]}...), unknown={len(subset.unknown_indices)}')
            m = submodels[i]
            opt = optim.SGD(m.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            sub_scheduler = MultiStepLR(opt, milestones=[40, 60], gamma=0.1)
            for ep in range(1, args.epochs_sub + 1):
                _, a = train_ce_epoch(m, loader, opt, device)
                sub_scheduler.step()
                current_lr = opt.param_groups[0]['lr']
                print(f'[Sub][{ep}] {group_names[i]} acc={a*100:.2f}%, lr={current_lr}')
                if ep % 10 == 0 or ep == args.epochs_sub:
                    acc_26 = eval_26class_acc(m, test_group_loaders[i], device)
                    acc_str = ', '.join([f'c{c}:{acc_26[c]:.1f}' for c in range(26)])
                    print(f'  [Sub][{ep}] {group_names[i]} 26-class acc: {acc_str}')
            torch.save(m.state_dict(), f'{args.model_dir}/wrn_{group_names[i]}_final.pt')
            m.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for m in submodels:
        m.to(device)
        for p in m.parameters():
            p.requires_grad = True

    T_init = args.route_T_start if (args.route_T_start != args.route_T_end) else args.route_T
    fusion = UnknownRoutingFusion100(submodels, group_fine, T=T_init, margin=args.route_margin).to(device)

    params = [{'params': m.parameters(), 'lr': args.lr} for m in fusion.submodels]
    optimizer = optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

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

    # CE warmup
    use_T_schedule = (args.route_T_start != args.route_T_end)
    if use_T_schedule:
        fusion.set_temperature(args.route_T_start)
        print(f'[INFO] Routing M1 (score=1-unk), T schedule: {args.route_T_start:.2f} -> {args.route_T_end:.2f}')
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

    freeze_bn_stats_only(fusion)
    print('[INFO] BN: freeze stats only (eval mode), gamma/beta remain trainable')

    if args.scheduler == 'step':
        milestones = [int(args.epochs_fusion * 0.4), int(args.epochs_fusion * 0.65)]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_fusion)
    else:
        scheduler = None

    if resume_loaded and checkpoint is not None and scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f'[RESUME] scheduler loaded, start_epoch={start_epoch}')

    print('==== TRADES training ====')
    bn_unfrozen = False
    for ep in range(start_epoch, args.epochs_fusion + 1):
        if use_T_schedule:
            t = (ep - 1) / max(1, args.epochs_fusion - 1)
            T_cur = args.route_T_start + (args.route_T_end - args.route_T_start) * t
            fusion.set_temperature(T_cur)

        fusion.train()
        if not bn_unfrozen:
            freeze_bn_stats_only(fusion)
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
            loss += aux_ce_loss_unk(aux_logits, y, group_defs, fine_to_coarse_t, weight=args.aux_weight, unknown_weight=args.aux_unknown_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / max(1, total_samples)
        current_lr = optimizer.param_groups[0]['lr']
        T_info = f', T={fusion.T:.2f}' if use_T_schedule else ''
        if ep % 5 == 0 or ep == args.epochs_fusion:
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
            print(f'[Fusion/TRADES][{ep}/{args.epochs_fusion}] test_acc={acc*100:.2f}%, train_loss={avg_loss:.4f}, lr={current_lr:.6f}{T_info}')
            if ep % 10 == 0 or ep == args.epochs_fusion:
                fusion.eval()
                with torch.no_grad():
                    ema.apply_to(fusion)
                    global_w, domain_ws = eval_routing_quality(fusion, test_group_loaders, test_loader, device, group_names)
                    gw_str = ', '.join([f'{group_names[i]}:{global_w[i]:.3f}' for i in range(4)])
                    dw_str = ', '.join([f'{group_names[i]}:{domain_ws[i]:.3f}' for i in range(4)])
                    print(f'  [Fusion][{ep}] global_w: {gw_str}')
                    print(f'  [Fusion][{ep}] domain_w (expert weight on own domain): {dw_str}')
                    for ex in range(4):
                        acc_26 = eval_26class_acc(fusion.submodels[ex], test_group_loaders[ex], device)
                        acc_str = ', '.join([f'c{c}:{acc_26[c]:.1f}' for c in range(26)])
                        print(f'  [Fusion][{ep}] expert {group_names[ex]} 26-class acc: {acc_str}')
                    ema.restore(fusion)
                fusion.train()
        else:
            print(f'[Fusion/TRADES][{ep}/{args.epochs_fusion}] train_loss={avg_loss:.4f}, lr={current_lr:.6f}{T_info}')

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
                    "group_fine": group_fine,
                    "args": vars(args),
                },
                f'{args.model_dir}/fusion-epoch{ep}.pt'
            )


if __name__ == '__main__':
    main()
