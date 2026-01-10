from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

from models.parallel_wrn import WRNWithEmbedding, ParallelFusionWRN


# =========================================================
# CIFAR-10 Normalize 
# =========================================================
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD White-box Attack (Parallel Fusion)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

# MUST match TRADES eval settings (in pixel space)
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--step-size', type=float, default=0.003)

# Use explicit store_true/store_false to avoid bool() pitfalls
parser.add_argument('--random', action='store_true',
                    help='use random initialization for PGD')
parser.add_argument('--no-random', dest='random', action='store_false',
                    help='disable random initialization for PGD')
parser.set_defaults(random=True)

# checkpoints
parser.add_argument('--model-path', required=True,
                    help='fusion checkpoint, e.g. model-parallel/fusion-epoch100.pt')
parser.add_argument('--wrn4-path', required=True,
                    help='4-class submodel, e.g. model-parallel/wrn4_final.pt')
parser.add_argument('--wrn6-path', required=True,
                    help='6-class submodel, e.g. model-parallel/wrn6_final.pt')

args = parser.parse_args()

use_cuda = (not args.no_cuda) and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


def _cifar_bounds(device_):
    """
    Bounds for normalized images:
      x_raw in [0,1]
      x_norm = (x_raw - mean)/std
      => x_norm in [(0-mean)/std, (1-mean)/std]
    """
    mean = torch.tensor(CIFAR10_MEAN, device=device_).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device_).view(1, 3, 1, 1)
    low = (0.0 - mean) / std
    high = (1.0 - mean) / std
    return low, high, std


# data loader (match training: ToTensor + Normalize)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random_start=True):
    """
    PGD in *normalized space* while epsilon/step_size are specified in pixel space.
    Returns: (err_natural, err_pgd) as scalars (#wrong in batch)
    """
    model.eval()

    # Convert eps/step from pixel space -> normalized space (per-channel)
    low, high, std = _cifar_bounds(X.device)
    eps = epsilon / std
    step = step_size / std

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    X_pgd = Variable(X.data, requires_grad=True)

    if random_start:
        random_noise = torch.empty_like(X_pgd).uniform_(-1.0, 1.0) * eps
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
        # project to valid range
        X_pgd = Variable(torch.max(torch.min(X_pgd, high), low), requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()

        eta = step * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # project to epsilon-ball around X (in normalized space)
        delta = torch.clamp(X_pgd.data - X.data, -eps, eps)
        X_pgd = Variable(X.data + delta, requires_grad=True)

        # clamp to valid normalized bounds
        X_pgd = Variable(torch.max(torch.min(X_pgd, high), low), requires_grad=True)

    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)

        err_natural, err_robust = pgd_whitebox(
            model, X, y,
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            random_start=args.random
        )
        natural_err_total += err_natural
        robust_err_total += err_robust

    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def main():
    # build models exactly as training
    m4 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=4).to(device)
    m6 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=6).to(device)

    # load submodels
    m4.load_state_dict(torch.load(args.wrn4_path, map_location=device))
    m6.load_state_dict(torch.load(args.wrn6_path, map_location=device))

    model = ParallelFusionWRN(m4, m6).to(device)

    # load fusion checkpoint
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print('pgd white-box attack (parallel fusion)')
    eval_adv_test_whitebox(model, test_loader)


if __name__ == '__main__':
    main()
