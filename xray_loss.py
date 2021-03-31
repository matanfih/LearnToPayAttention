import torch
import torch.nn as nn
from torch import Tensor


def PALLoss(Y, f):
    v = -Y * torch.log(torch.exp(f) / torch.sum(torch.exp(f)))
    return torch.sum(v)


def PALLoss_N(Y, f):
    v = -Y / torch.sum(Y) * torch.log(torch.exp(f) / torch.sum(torch.exp(f)))
    return torch.sum(v)


def OVALoss(Y, f):
    U = torch.exp(f)
    D = (1 + torch.exp(f))

    v = -Y * torch.log(U / D) - (1 - Y) * torch.log(1 / D)
    #print("U=%s, D=%s, Y=%s" % (U, D, Y))
    return torch.sum(v)


def OVALoss_N(Y, f):
    v = -(Y / torch.sum(Y)) * torch.log(torch.exp(f) / (1 + torch.exp(f))) \
        - (1 - Y / torch.sum(Y)) * torch.log(1 / (1 + torch.exp(f)))
    return torch.sum(v)


class Loss(nn.modules.loss.Module):
    reduction: str

    def __init__(self, loss='OVA') -> None:
        super(Loss, self).__init__()
        if loss == 'OVA':
            self.loss = OVALoss
        elif loss == 'OVA_N':
            self.loss = OVALoss_N
        elif loss == 'PAL':
            self.loss = PALLoss
        elif loss == 'PAL_N':
            self.loss = PALLoss_N

        assert self.loss is not None, "loss {} not found".format(loss)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss(input, target)


