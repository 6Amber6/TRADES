import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def _normalize_bounds(mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    low = (0.0 - mean_t) / std_t
    high = (1.0 - mean_t) / std_t
    return low, high, std_t


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf',
                data_mean=None,
                data_std=None,
                return_x_adv=False):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    was_training = model.training
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    if data_mean is not None and data_std is not None:
        low, high, std = _normalize_bounds(data_mean, data_std, x_natural.device)
        step = step_size / std
        eps = epsilon / std
    else:
        low, high = 0.0, 1.0
        step = step_size
        eps = epsilon

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - eps), x_natural + eps)
            x_adv = torch.max(torch.min(x_adv, high), low)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            if isinstance(low, torch.Tensor) or isinstance(high, torch.Tensor):
                delta.data = torch.max(torch.min(delta.data, high), low).sub_(x_natural)
            else:
                delta.data.clamp_(low, high).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        if isinstance(low, torch.Tensor) or isinstance(high, torch.Tensor):
            x_adv = torch.max(torch.min(x_adv, high), low)
        else:
            x_adv = torch.clamp(x_adv, low, high)
    if was_training:
        model.train()

    if isinstance(low, torch.Tensor) or isinstance(high, torch.Tensor):
        x_adv = Variable(torch.max(torch.min(x_adv, high), low), requires_grad=False)
    else:
        x_adv = Variable(torch.clamp(x_adv, low, high), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    if return_x_adv:
        return loss, x_adv
    return loss
