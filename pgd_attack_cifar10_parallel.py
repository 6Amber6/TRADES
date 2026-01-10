from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

from models.parallel_wrn import WRNWithEmbedding, ParallelFusionWRN


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD White-box Attack (Parallel Fusion)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

# MUST match TRADES eval settings
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--step-size', type=float, default=0.003)

# Use explicit store_true/store_false to avoid bool() pitfalls
parser.add_argument('--random', action='store_true',
                    help='use random initialization for PGD (default: False)')
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

# data loader (same as TRADES pgd_attack_cifar10.py)
transform_test = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random_start=True):
    """
    Returns: (err_natural, err_pgd) as scalars (#wrong in batch)
    """
    model.eval()

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    X_pgd = Variable(X.data, requires_grad=True)

    if random_start:
        random_noise = torch.empty_like(X_pgd).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)

        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err.item(), err_pgd.item()


def eval_adv_test_whitebox(model, device, loader):
    model.eval()
    robust_err_total = 0.0
    natural_err_total = 0.0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)

        err_nat, err_pgd = pgd_whitebox(
            model, X, y,
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            random_start=args.random
        )

        natural_err_total += err_nat
        robust_err_total += err_pgd
        total += data.size(0)

    print('natural_err_total:', natural_err_total, '/', total)
    print('robust_err_total:', robust_err_total, '/', total)
    print('natural_err_rate:', natural_err_total / total)
    print('robust_err_rate:', robust_err_total / total)


def main():
    # build models exactly as training
    m4 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=4).to(device)
    m6 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=6).to(device)

    # load submodels
    m4.load_state_dict(torch.load(args.wrn4_path, map_location=device))
    m6.load_state_dict(torch.load(args.wrn6_path, map_location=device))

    model = ParallelFusionWRN(m4, m6).to(device)

    # load fusion checkpoint (contains fc, and may also include m4/m6; loading is safe)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print('pgd white-box attack (parallel fusion)')
    eval_adv_test_whitebox(model, device, test_loader)


if __name__ == '__main__':
    main()
