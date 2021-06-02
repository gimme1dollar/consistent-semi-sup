# Implemented upon pytorch 1.2.0
import torch.nn as nn
import torch
import torchvision.transforms.functional as F
from utils.visualize import *

def CEloss(inputs, gt, ignore_index=255):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)(inputs, gt)

def MSEloss(inputs, gt):
    c = nn.MSELoss()
    loss = c(inputs,gt)
    return loss

def total_loss(losses_list):
    total = 0
    for component in losses_list:
        if isinstance(component, list):
            total += sum(component)
        else:
            total += component
    return total

def softmax_kl_loss(input_logits, target_logits):
    input_log_softmax = F.log_softmax(input_logits, dim=0)
    target_softmax = F.softmax(target_logits, dim=0)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)

# bring from https://github.com/jik0730/VAT-pytorch
class VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, model, n_power, XI, eps):
        super(VAT, self).__init__()
        self.model = model
        self.n_power = n_power
        self.XI = XI
        self.epsilon = eps

    def forward(self, X, logit):
        vat_loss = virtual_adversarial_loss(X, logit, self.model, self.n_power,
                                            self.XI, self.epsilon)
        return vat_loss  # already averaged


def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    return qlogq - qlogp


def get_normalized_vector(d):
    d_abs_max = torch.max(
        torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(
            d.size(0), 1, 1, 1)
    # print(d_abs_max.size())
    d /= (1e-12 + d_abs_max)
    d /= torch.sqrt(1e-6 + torch.sum(
        torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True))
    return d


def generate_virtual_adversarial_perturbation(x, logit, model, n_power, XI,
                                              epsilon):
    d = torch.randn_like(x)

    for _ in range(n_power):
        d = XI * get_normalized_vector(d).requires_grad_()
        logit_m = model(x + d)
        dist = kl_divergence_with_logit(logit, logit_m)
        grad = torch.autograd.grad(dist, [d])[0]
        d = grad.detach()

    return epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, model, n_power, XI, epsilon):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, model,
                                                       n_power, XI, epsilon)
    logit_p = logit.detach()
    logit_m = model(x + r_vadv)
    loss = kl_divergence_with_logit(logit_p, logit_m)
    return loss