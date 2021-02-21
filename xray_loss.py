import torch.nn as nn
#nn.CrossEntropyLoss()
import torch


def PALLoss(Y, f):
    v = -Y * torch.log(torch.exp(f) / torch.sum(torch.exp(f)))
    return torch.sum(v)

def PALLoss_N(Y, f):
    v = -Y / torch.sum(Y) * torch.log(torch.exp(f) / torch.sum(torch.exp(f)))
    return torch.sum(v)

def OVALoss(Y, f):
    v = -Y*torch.log(torch.exp(f) / (1 + torch.exp(f))) - (1 - Y) * torch.log(1 / (1 + torch.exp(f)))
    return torch.sum(v)

def OVALoss_N(Y, f):
    v = -(Y / torch.sum(Y)) * torch.log(torch.exp(f) / (1 + torch.exp(f))) \
        - (1 - Y / torch.sum(Y)) * torch.log(1 / (1 + torch.exp(f)))
    return torch.sum(v)
