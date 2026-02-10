from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import transforms

from models.parallel_wrn import WRNWithEmbedding


parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 PGD Attack Evaluation (Parallel Fusion)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--step-size', type=float, default=0.003)
parser.add_argument('--random', default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./model-cifar100-parallel/fusion-epoch200.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--sub-depth', type=int, default=28, choices=[28, 34])
parser.add_argument('--sub-widen', type=int, default=10, choices=[8, 10])

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

transform_test = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


class ParallelFusionWRN100(nn.Module):
    def __init__(self, submodels, num_classes=100):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        emb_dim = sum(m.nChannels for m in self.submodels)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        embs = []
        for m in self.submodels:
            emb, _ = m(x, return_embedding=True)
            embs.append(emb)
        combined = torch.cat(embs, dim=1)
        return self.fc(combined)


def build_fusion_model():
    submodels = [
        WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen, num_classes=25).to(device)
        for _ in range(4)
    ]
    model = ParallelFusionWRN100(submodels, num_classes=100).to(device)
    return model


def _pgd_whitebox(model, X, y, epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

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
    print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def main():
    if args.white_box_attack:
        print('pgd white-box attack')
        model = build_fusion_model()
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        eval_adv_test_whitebox(model, device, test_loader)
    else:
        raise ValueError('Only white-box attack is supported for fusion model evaluation.')


if __name__ == '__main__':
    main()
