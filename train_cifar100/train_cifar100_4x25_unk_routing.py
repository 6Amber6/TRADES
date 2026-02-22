# CIFAR-100: 4 experts x 25 classes, each with unknown, routing by score = a*(1-unk_self) + b*unk_others

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
# Balanced batch sampler: 1:1 known vs unknown per batch
# Uses max(n_known, n_unknown) batches, oversamples shorter side to cover all data
# =========================================================
class BalancedKnownUnknownSampler(torch.utils.data.Sampler):
    def __init__(self, known_indices, unknown_indices, batch_size, shuffle=True):
        self.known_indices = list(known_indices)
        self.unknown_indices = list(unknown_indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_known = len(self.known_indices)
        self.n_unknown = len(self.unknown_indices)
        n_per_half = batch_size // 2
        self.n_batches = max(self.n_known, self.n_unknown) // n_per_half
        self.n_batches = max(1, self.n_batches)

    def __iter__(self):
        import random
        n_per_half = self.batch_size // 2
        known = self.known_indices.copy()
        unknown = self.unknown_indices.copy()
        if self.shuffle:
            random.shuffle(known)
            random.shuffle(unknown)
        for b in range(self.n_batches):
            batch = []
            for i in range(n_per_half):
                idx = b * n_per_half + i
                batch.append(known[idx % self.n_known])
                batch.append(unknown[idx % self.n_unknown])
            if self.shuffle:
                random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# =========================================================
# BN freeze/unfreeze
# freeze_bn_stats_only: eval() only, keep gamma/beta trainable (better capacity)
# freeze_bn: full freeze (eval + no grad)
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
# Fusion: scatter + weighted. Non-owned classes contribute 0.
# (Avoid -1e4 mask: w_j * (-1e4) would bias logits and cause routing collapse)
# =========================================================


# =========================================================
# Unknown-based routing: score_i = a*(1-unk_i) + b*sum(unk_j, j!=i)
# =========================================================
def unknown_routing_scores(unk_probs, a=1.0, b=0.5, T=1.0):
    """
    unk_probs: [B, 4] each expert's unknown probability
    score_i = a*(1-unk_i) + b * mean(unk_j for j!=i)
    """
    B = unk_probs.size(0)
    n = unk_probs.size(1)
    conf = 1 - unk_probs
    other_unk = unk_probs.unsqueeze(2).expand(B, n, n)
    mask = 1 - torch.eye(n, device=unk_probs.device).unsqueeze(0).expand(B, n, n)
    other_unk = (other_unk * mask).sum(2) / (n - 1)
    scores = a * conf + b * other_unk
    w = F.softmax(scores / T, dim=1)
    return w


# =========================================================
# Fusion model: 4 experts (25+1 each), unknown-based routing
# =========================================================
class UnknownRoutingFusion100(nn.Module):
    def __init__(self, submodels, owned_fine_per_group, a=1.0, b=0.5, T=1.0, margin=0.0):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        self.owned_fine = owned_fine_per_group
        self.a = a
        self.b = b
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
        w = unknown_routing_scores(unk_probs, a=self.a, b=self.b, T=self.T)

        B = group_logits[0].size(0)
        device = group_logits[0].device
        final = torch.zeros(B, 100, device=device, dtype=group_logits[0].dtype)
        for i, (loc, owned) in enumerate(zip(group_logits, self.owned_fine)):
            known = loc[:, :len(owned)]
            # Scale by std only: align logit magnitude across experts
            std = known.std(dim=1, keepdim=True).clamp(min=1e-6)
            known = known / std
            final[:, owned] += w[:, i:i+1] * known

        if return_aux:
            return group_logits, final
        return final


# =========================================================
# Aux loss: positive (known) + negative (unknown) supervision
# Each expert must learn: in-domain -> correct class; out-of-domain -> Unknown
# unknown_weight < 1: avoid experts over-predicting unknown (out-domain 75 classes, batch占比高)
# =========================================================
def aux_ce_loss_unk(group_logits, y, group_defs, fine_to_coarse_t, weight=0.1, unknown_weight=0.3):
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
parser = argparse.ArgumentParser(description='CIFAR-100 4x25+unk routing TRADES')
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
parser.add_argument('--aux-weight', type=float, default=0.05)
parser.add_argument('--aux-unknown-weight', type=float, default=0.3,
                    help='down-weight out-of-domain->Unknown loss; avoid experts over-predicting unknown')
parser.add_argument('--route-a', type=float, default=1.0)
parser.add_argument('--route-b', type=float, default=0.5)
parser.add_argument('--route-T', type=float, default=1.0, help='fixed T if route-T-end not set')
parser.add_argument('--route-T-start', type=float, default=1.0, help='T at epoch 1 (less smoothing than 2.0, preserve expert signal)')
parser.add_argument('--route-T-end', type=float, default=0.2, help='T at last epoch (sharper routing); set both for schedule')
parser.add_argument('--route-margin', type=float, default=0.0)
parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'none'])
parser.add_argument('--model-dir', default='./model-cifar100-4x25-unk')
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

    base_train_sub = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_sub)
    base_train_fusion = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_fusion)
    test_set = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)

    group_names = GROUP_ORDER
    group_coarse = [COARSE_GROUPS[name] for name in group_names]
    group_fine = [build_fine_classes_for_group(g) for g in group_coarse]

    group_loaders = []
    for fine_classes in group_fine:
        subset = CIFARUnknownSubset(base_train_sub, fine_classes)
        sampler = BalancedKnownUnknownSampler(
            subset.known_indices, subset.unknown_indices,
            batch_size=args.batch_size, shuffle=True
        )
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
        print('==== Stage 1 (4x25+unk pretrain, 1:1 known/unknown per batch) ====')
        for i, loader in enumerate(group_loaders):
            subset = loader.dataset
            print(f'  {group_names[i]}: known={len(subset.known_indices)}, unknown={len(subset.unknown_indices)}')
            m = submodels[i]
            opt = optim.SGD(m.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for ep in range(1, args.epochs_sub + 1):
                _, a = train_ce_epoch(m, loader, opt, device)
                print(f'[Sub][{ep}] {group_names[i]} acc={a*100:.2f}%')
            torch.save(m.state_dict(), f'{args.model_dir}/wrn_{group_names[i]}_final.pt')
            m.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for m in submodels:
        m.to(device)
        for p in m.parameters():
            p.requires_grad = True

    T_init = args.route_T_start if (args.route_T_start != args.route_T_end) else args.route_T
    fusion = UnknownRoutingFusion100(submodels, group_fine, a=args.route_a, b=args.route_b, T=T_init, margin=args.route_margin).to(device)

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

    # CE warmup (use T_start for smoother routing)
    use_T_schedule = (args.route_T_start != args.route_T_end)
    if use_T_schedule:
        fusion.set_temperature(args.route_T_start)
        print(f'[INFO] Routing T schedule: {args.route_T_start:.2f} -> {args.route_T_end:.2f}')
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
        milestones = [int(args.epochs_fusion * 0.5), int(args.epochs_fusion * 0.75)]
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
