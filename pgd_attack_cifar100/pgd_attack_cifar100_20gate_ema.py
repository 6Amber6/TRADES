from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import transforms

from models.parallel_wrn import WRNWithEmbedding


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
# Utils
# =========================================================
def _extract_logits(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class SoftGateFusionWRN100(nn.Module):
    def __init__(self, experts, fine_groups, gate):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        for i, g in enumerate(fine_groups):
            self.register_buffer(f"fine_idx_{i}", torch.tensor(g, dtype=torch.long))

    def forward(self, x):
        gate_logits = _extract_logits(self.gate(x))
        gate_log_probs = torch.nn.functional.log_softmax(gate_logits, dim=1)
        batch_size = x.size(0)
        device = x.device
        final_logits = torch.full((batch_size, 100), -1e9, device=device)
        for i, expert in enumerate(self.experts):
            logits_i = _extract_logits(expert(x))
            log_g_i = gate_log_probs[:, i:i + 1]
            fine_idx = getattr(self, f"fine_idx_{i}")
            final_logits[:, fine_idx] = log_g_i + logits_i
        return final_logits


def pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random_start=True):
    model.eval()
    err = (model(X).argmax(1) != y).float().sum().item()

    X_pgd = Variable(X.data, requires_grad=True)
    if random_start:
        noise = torch.empty_like(X_pgd).uniform_(-1.0, 1.0) * epsilon
        X_pgd = Variable(X_pgd.data + noise, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0.0, 1.0), requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        delta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(torch.clamp(X.data + delta, 0.0, 1.0), requires_grad=True)

    err_pgd = (model(X_pgd).argmax(1) != y).float().sum().item()
    return err, err_pgd


def eval_adv_test_whitebox(model, loader, epsilon, num_steps, step_size, random_start=True):
    natural_err_total = 0
    robust_err_total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        err_nat, err_adv = pgd_whitebox(
            model, x, y,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            random_start=random_start
        )
        natural_err_total += err_nat
        robust_err_total += err_adv
    natural_acc = 1.0 - natural_err_total / len(loader.dataset)
    robust_acc = 1.0 - robust_err_total / len(loader.dataset)
    print('natural_acc: {:.4f}, robust_acc: {:.4f}'.format(natural_acc, robust_acc))


parser = argparse.ArgumentParser(description='PGD White-box Attack (CIFAR-100 20-gate)')
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--step-size', type=float, default=0.003)
parser.add_argument('--random', action='store_true')
parser.add_argument('--no-random', dest='random', action='store_false')
parser.set_defaults(random=True)
parser.add_argument('--model-path', required=True)
parser.add_argument('--ema', action='store_true',
                    help='evaluate using EMA shadow in checkpoint')

args = parser.parse_args()
use_cuda = (not args.no_cuda) and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


def main():
    # data (pixel space, no Normalize)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR100(
        root='../data', train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    # build coarse->fine groups (20 experts, 5 fine classes each)
    coarse_to_fine = [[] for _ in range(20)]
    for fine_id, coarse_id in enumerate(CIFAR100_FINE_TO_COARSE):
        coarse_to_fine[coarse_id].append(fine_id)

    experts = []
    for fine_ids in coarse_to_fine:
        m = WRNWithEmbedding(depth=16, widen_factor=4, num_classes=len(fine_ids)).to(device)
        experts.append(m)

    gate = WRNWithEmbedding(depth=16, widen_factor=4, num_classes=20).to(device)
    model = SoftGateFusionWRN100(experts, coarse_to_fine, gate).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        if args.ema:
            shadow = ckpt.get('ema_shadow')
            if shadow is not None:
                for n, p in model.named_parameters():
                    if n in shadow:
                        p.data.copy_(shadow[n].to(device))
    else:
        model.load_state_dict(ckpt)

    eval_adv_test_whitebox(
        model, test_loader,
        epsilon=args.epsilon,
        num_steps=args.num_steps,
        step_size=args.step_size,
        random_start=args.random
    )


if __name__ == '__main__':
    main()
