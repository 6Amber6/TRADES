# WideResNet 16-4 * 20
# gating
# unknown class or not (use_unknown=True/False)
# Shared Backbone

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

from models.wideresnet_update import BasicBlock, NetworkBlock
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
# Dataset wrapper with unknown class
# =========================================================
class CIFARCoarseUnknown(torch.utils.data.Dataset):
    def __init__(self, base_dataset, fine_ids, unknown_idx):
        self.base = base_dataset
        self.fine_ids = list(fine_ids)
        self.unknown_idx = unknown_idx
        self.map = {c: i for i, c in enumerate(self.fine_ids)}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        y = int(y)
        if y in self.map:
            return x, self.map[y]
        return x, self.unknown_idx


# =========================================================
# BN freeze/unfreeze util
# =========================================================
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


def unfreeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()


# =========================================================
# Shared WRN with 20 heads + gate
# =========================================================
class SharedWRN20Gate(nn.Module):
    def __init__(self, depth=34, widen_factor=10, num_heads=20, num_known=5, unknown_idx=5,
                 gate_hidden=128, dropRate=0.0, fine_groups=None, n_block3=2, use_unknown=True):
        super().__init__()
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.num_heads = num_heads
        self.num_known = num_known
        self.unknown_idx = unknown_idx
        self.use_unknown = use_unknown

        # Shared backbone
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], BasicBlock, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], BasicBlock, 2, dropRate)

        # Heads with reduced block3
        self.block3 = nn.ModuleList([
            NetworkBlock(n_block3, nChannels[2], nChannels[3], BasicBlock, 2, dropRate)
            for _ in range(num_heads)
        ])
        self.bn = nn.ModuleList([nn.BatchNorm2d(nChannels[3]) for _ in range(num_heads)])
        out_dim = num_known + (1 if self.use_unknown else 0)
        self.fc = nn.ModuleList([nn.Linear(nChannels[3], out_dim) for _ in range(num_heads)])

        # Gate on shared features
        self.gate_fc1 = nn.Linear(nChannels[2], gate_hidden)
        self.gate_fc2 = nn.Linear(gate_hidden, num_heads)

        # Register fine indices for fusion
        if fine_groups is None:
            fine_groups = [[] for _ in range(num_heads)]
        for i, g in enumerate(fine_groups):
            self.register_buffer(f"fine_idx_{i}", torch.tensor(g, dtype=torch.long))

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / k) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def shared_parameters(self):
        return list(self.conv1.parameters()) + list(self.block1.parameters()) + list(self.block2.parameters())

    def head_parameters(self, i):
        return list(self.block3[i].parameters()) + list(self.bn[i].parameters()) + list(self.fc[i].parameters())

    def gate_parameters(self):
        return list(self.gate_fc1.parameters()) + list(self.gate_fc2.parameters())

    def _gate(self, shared_feat):
        pooled = F.adaptive_avg_pool2d(shared_feat, 1).view(shared_feat.size(0), -1)
        return self.gate_fc2(F.relu(self.gate_fc1(pooled)))

    def _head_forward(self, shared_feat, i):
        out = self.block3[i](shared_feat)
        out = self.relu(self.bn[i](out))
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return self.fc[i](out)

    def forward(self, x, return_aux=False, return_gate=False, head=None):
        shared = self.conv1(x)
        shared = self.block1(shared)
        shared = self.block2(shared)

        gate_logits = self._gate(shared)  # (N, 20)

        if head is not None:
            return self._head_forward(shared, head)

        # Fusion to 100-way logits
        gate_log_probs = F.log_softmax(gate_logits, dim=1)
        batch_size = x.size(0)
        final_logits = x.new_full((batch_size, 100), -1e9)
        aux_logits = []
        for i in range(self.num_heads):
            logits_i = self._head_forward(shared, i)
            aux_logits.append(logits_i)
            fine_idx = getattr(self, f"fine_idx_{i}")
            log_g_i = gate_log_probs[:, i:i + 1]
            # known logits (length = number of fine ids for this head)
            known_logits = logits_i[:, :len(fine_idx)]
            if self.use_unknown:
                # Use unknown logit as a confidence offset for known classes
                unk_logit = logits_i[:, self.unknown_idx:self.unknown_idx + 1]
                final_logits[:, fine_idx] = log_g_i + (known_logits - unk_logit)
            else:
                # No unknown output: directly fuse using known logits
                final_logits[:, fine_idx] = log_g_i + known_logits

        if return_aux and return_gate:
            return aux_logits, final_logits, gate_logits
        if return_aux:
            return aux_logits, final_logits
        if return_gate:
            return final_logits, gate_logits
        return final_logits


# =========================================================
# Loss helpers
# =========================================================
def aux_ce_loss(all_logits, y, map_t_list, weight=0.02, unknown_idx=5, unknown_weight=0.1, use_unknown=True):
    loss = 0.0
    for i, logits in enumerate(all_logits):
        map_t = map_t_list[i]
        # y_local: shape [B], values in {0..K-1} or -1 (when use_unknown=False)
        y_local = map_t[y]

        if use_unknown:
            # compute per-sample CE and down-weight unknown samples
            ce = F.cross_entropy(logits, y_local, reduction='none')
            # make weight tensor
            w = torch.ones_like(ce)
            mask_unknown = (y_local == unknown_idx)
            if mask_unknown.any():
                w = w.to(ce.device)
                w[mask_unknown] = float(unknown_weight)
            loss = loss + (w * ce).mean()
        else:
            # skip samples not belonging to this head (mapped as -1)
            valid_mask = (y_local >= 0)
            if valid_mask.sum() == 0:
                continue
            logits_valid = logits[valid_mask]
            y_valid = y_local[valid_mask]
            ce = F.cross_entropy(logits_valid, y_valid)
            loss = loss + ce

    return weight * loss


def gate_supervision_loss(gate_logits, coarse_y):
    return F.cross_entropy(gate_logits, coarse_y)


def gate_kl_loss(gate_adv, gate_clean, eps=1e-6):
    log_p = F.log_softmax(gate_adv, dim=1)
    q = F.softmax(gate_clean, dim=1).clamp(eps, 1 - eps)
    return F.kl_div(log_p, q, reduction='batchmean')


# =========================================================
# Arguments
# =========================================================
parser = argparse.ArgumentParser(description='TRADES CIFAR-100 Shared WRN (20 heads + unknown)')
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
parser.add_argument('--n-block3', type=int, default=2, help='Number of BasicBlocks in block3 per head')
parser.add_argument('--use-unknown', action='store_true', default=True, help='Whether to use unknown class')
parser.add_argument('--no-unknown', dest='use_unknown', action='store_false', help='Disable unknown class')
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file for resuming')
parser.add_argument('--model-dir', default='./model-cifar100-shared-20u')
args = parser.parse_args()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_unknown = args.use_unknown
    unknown_weight_stage1 = 0.1 if use_unknown else 1.0
    unknown_weight_stage2 = 0.1 if use_unknown else 1.0
    
    # 根据 use_unknown 调整模型目录名
    model_dir = args.model_dir + ('_with_unknown' if use_unknown else '_no_unknown')
    os.makedirs(model_dir, exist_ok=True)
    print(f'[INFO] Using unknown class: {use_unknown}')
    print(f'[INFO] Model directory: {model_dir}')

    # Build coarse -> fine groups
    coarse_to_fine = [[] for _ in range(20)]
    for fine_id, coarse_id in enumerate(CIFAR100_FINE_TO_COARSE):
        coarse_to_fine[coarse_id].append(fine_id)

    # Precompute mapping for aux loss
    # - use_unknown=True: map unknown classes to index=len(fine_ids)
    # - use_unknown=False: map non-known classes to -1 (they will be skipped)
    map_t_list = []
    for fine_ids in coarse_to_fine:
        if use_unknown:
            map_t = torch.full((100,), len(fine_ids), dtype=torch.long, device=device)
            map_t[torch.tensor(fine_ids, device=device)] = torch.arange(len(fine_ids), device=device)
        else:
            map_t = torch.full((100,), -1, dtype=torch.long, device=device)
            map_t[torch.tensor(fine_ids, device=device)] = torch.arange(len(fine_ids), device=device)
        map_t_list.append(map_t)
    fine_to_coarse_t = torch.tensor(CIFAR100_FINE_TO_COARSE, device=device, dtype=torch.long)

    # Data
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    base_train = torchvision.datasets.CIFAR100(
        root='../data', train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root='../data', train=False, download=True, transform=transform
    )

    train_loader_full = torch.utils.data.DataLoader(
        base_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    # Per-head loaders (known + unknown)
    head_loaders = []
    for fine_ids in coarse_to_fine:
        if use_unknown:
            ds = CIFARCoarseUnknown(base_train, fine_ids, unknown_idx=len(fine_ids))
        else:
            # 不使用unknown class：只用已知类数据
            ds = CIFARCoarseUnknown(base_train, fine_ids, unknown_idx=len(fine_ids))
            # 但通过设置自定义数据集过滤
            class CIFARCoarseKnownOnly(torch.utils.data.Dataset):
                def __init__(self, base_dataset, fine_ids):
                    self.base = base_dataset
                    self.fine_ids = list(fine_ids)
                    self.map = {c: i for i, c in enumerate(self.fine_ids)}
                    # 过滤数据集
                    self.valid_indices = [i for i in range(len(base_dataset)) 
                                         if base_dataset[i][1] in self.fine_ids]
                
                def __len__(self):
                    return len(self.valid_indices)
                
                def __getitem__(self, i):
                    x, y = self.base[self.valid_indices[i]]
                    y = int(y)
                    return x, self.map[y]
            
            ds = CIFARCoarseKnownOnly(base_train, fine_ids)
        
        head_loaders.append(torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
        ))

    # ================= Stage 1 =================
    fusion = SharedWRN20Gate(depth=34, widen_factor=10, num_heads=20,
                             num_known=5, unknown_idx=5, fine_groups=coarse_to_fine,
                             n_block3=args.n_block3, use_unknown=use_unknown).to(device)

    stage1_params = fusion.shared_parameters()
    for i in range(20):
        stage1_params += fusion.head_parameters(i)
    opt_stage1 = optim.SGD(stage1_params, lr=args.lr,
                           momentum=args.momentum, weight_decay=args.weight_decay)

    print('==== Stage 1 ====')
    for ep in range(1, args.epochs_sub + 1):
        fusion.train()
        acc_list = []
        for i in range(20):
            total, correct = 0, 0
            for x, y in head_loaders[i]:
                x, y = x.to(device), y.to(device)
                opt_stage1.zero_grad()
                logits = fusion(x, head=i)
                
                if use_unknown:
                    # 加权CE损失：unknown类权重较小
                    sample_weights = torch.where(
                        y == len(coarse_to_fine[i]),  # unknown类的索引
                        torch.tensor(unknown_weight_stage1, device=y.device),
                        torch.tensor(1.0, device=y.device)
                    )
                    ce = F.cross_entropy(logits, y, reduction='none')
                    loss = (sample_weights * ce).mean()
                else:
                    # 标准CE损失（无unknown class）
                    loss = F.cross_entropy(logits, y)
                
                loss.backward()
                opt_stage1.step()
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            acc_list.append(correct / max(1, total))
        print(f'[Sub][{ep}] avg acc={sum(acc_list)/len(acc_list)*100:.2f}%')

    torch.save(fusion.state_dict(), f'{model_dir}/shared_stage1_final.pt')

    # ================= Stage 2 =================
    params = []
    for i in range(20):
        params.append({'params': fusion.head_parameters(i), 'lr': args.lr})
    params.append({'params': fusion.gate_parameters(), 'lr': args.lr})
    params.append({'params': fusion.shared_parameters(), 'lr': args.lr * 0.2})

    optimizer = optim.SGD(
        params,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    ema = EMA(fusion, decay=0.999)
    start_epoch = 1
    best_acc = 0.0
    
    # 断点续训：加载checkpoint
    if args.resume and args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f'[INFO] Loading checkpoint from {args.checkpoint}...')
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # 加载模型权重
            if 'model_state' in checkpoint:
                fusion.load_state_dict(checkpoint['model_state'])
                print('[INFO] Model state loaded')
            
            # 加载优化器状态
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                print('[INFO] Optimizer state loaded')
            
            # 加载EMA状态
            if 'ema_state' in checkpoint:
                ema.shadow = checkpoint['ema_state']
                print('[INFO] EMA state loaded')
            
            # 加载训练状态
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f'[INFO] Resuming from epoch {start_epoch}')
            
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                print(f'[INFO] Best accuracy so far: {best_acc*100:.2f}%')
        else:
            print(f'[WARNING] Checkpoint not found: {args.checkpoint}')
    elif args.resume:
        # 自动查找最近的checkpoint
        last_model = f'{model_dir}/last_model.pt'
        ckpt_path = f'{model_dir}/checkpoint_latest.pt'
        if os.path.exists(ckpt_path):
            print(f'[INFO] Auto-loading checkpoint from {ckpt_path}...')
            checkpoint = torch.load(ckpt_path, map_location=device)
            fusion.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            ema.shadow = checkpoint['ema_state']
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            print(f'[INFO] Resuming from epoch {start_epoch}')
        else:
            print(f'[INFO] No checkpoint found, starting from scratch')

    # -------- CE warmup (10 epochs, BN NOT frozen) --------
    if start_epoch == 1:
        print('==== CE warmup (10 epochs) ====')
        for ep in range(1, 11):
            fusion.train()
            for x, y in train_loader_full:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = fusion(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
                optimizer.step()
                ema.update(fusion)
            print(f'[Warmup] Epoch {ep}/10, lr={optimizer.param_groups[0]["lr"]:.6f}')
    else:
        print(f'[Warmup] Skipping CE warmup because resuming from epoch {start_epoch}')

    # -------- Freeze BN AFTER warmup --------
    freeze_bn(fusion)

    # -------- TRADES training with MultiStepLR --------
    milestones = [args.epochs_fusion // 2, int(args.epochs_fusion * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    print('==== TRADES training ====')
    bn_unfrozen = False
    gate_sup_weight = 0.1
    gate_stab_weight = 0.1
    save_interval = 10  # 每10个epoch保存一次
    for ep in range(start_epoch, args.epochs_fusion + 1):
        fusion.train()
        if not bn_unfrozen:
            freeze_bn(fusion)

        total, correct = 0, 0
        for x, y in train_loader_full:
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
                data_mean=CIFAR100_MEAN,
                data_std=CIFAR100_STD,
                return_x_adv=True
            )

            aux_adv, _, gate_adv = fusion(x_adv, return_aux=True, return_gate=True)
            # auxiliary per-head CE (handles both use_unknown True/False)
            loss += aux_ce_loss(aux_adv, y, map_t_list, weight=0.02, unknown_weight=unknown_weight_stage2, use_unknown=use_unknown)

            _, _, gate_clean = fusion(x, return_aux=True, return_gate=True)
            coarse_y = fine_to_coarse_t[y]
            loss += gate_sup_weight * gate_supervision_loss(gate_clean, coarse_y)
            loss += gate_stab_weight * gate_kl_loss(gate_adv, gate_clean.detach())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            ema.update(fusion)

            with torch.no_grad():
                ema.apply_to(fusion)
                pred = fusion(x).argmax(1)
                ema.restore(fusion)
                correct += (pred == y).sum().item()
                total += y.size(0)

        if not bn_unfrozen and ep >= 40:
            unfreeze_bn(fusion)
            bn_unfrozen = True
            print(f'[INFO] Unfroze BN at epoch {ep}')

        scheduler.step()
        acc = correct / max(1, total)
        print(f'[Fusion/TRADES][{ep}/{args.epochs_fusion}] acc={acc*100:.2f}%, lr={optimizer.param_groups[0]["lr"]:.6f}')

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            with torch.no_grad():
                ema.apply_to(fusion)
                torch.save(fusion.state_dict(), f'{model_dir}/best_ema_model.pt')
                ema.restore(fusion)
            print(f'[SAVE] Best EMA model saved at epoch {ep} with acc={acc*100:.2f}%')

        # 每10个epoch保存模型
        if ep % save_interval == 0:
            torch.save(fusion.state_dict(), f'{model_dir}/milestone-epoch{ep}.pt')
            print(f'[SAVE] Milestone model saved at epoch {ep}')

        # 总是保存最后一个模型和完整checkpoint
        torch.save(fusion.state_dict(), f'{model_dir}/last_model.pt')
        
        # 保存完整checkpoint用于断点续训
        checkpoint_state = {
            'epoch': ep,
            'model_state': fusion.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'ema_state': ema.shadow,
            'best_acc': best_acc,
            'args': args
        }
        torch.save(checkpoint_state, f'{model_dir}/checkpoint_latest.pt')


if __name__ == '__main__':
    main()
