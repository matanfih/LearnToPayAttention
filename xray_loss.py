import torch
import torch.nn as nn
from torch import Tensor


def Y_N(Y):
    Y_sum = torch.sum(Y, dim=1, keepdim=True)
    Y_N = Y / Y_sum  #.reshape(-1, 1)
    Y_N[Y_N != Y_N] = 0
    #print("y_n: Y:%s, Y_sum:%s, Y_N:%s" % (Y, Y_sum, Y_N))
    return Y_N


def PALLoss(f, Y):
    U = torch.exp(f)
    v = -Y * torch.log(U / torch.sum(U, dim=1, keepdim=True))
    return torch.sum(v)


def PALLoss_N(f, Y):
    U = torch.exp(f)
    #print("y_n: Y:%s, Y_N:%s, U:%s" % (Y, Y_N(Y), U))
    v = -Y_N(Y) * torch.log(U / torch.sum(U, dim=1, keepdim=True))
    return torch.sum(v)


def OVALoss(f, Y):
    U = torch.exp(f)
    D = (1 + torch.exp(f))
    v = -Y * torch.log(U / D) - (1 - Y) * torch.log(1 / D)
    #print("U=%s, D=%s, Y=%s" % (U, D, Y))
    return torch.sum(v)


def OVALoss_N(f, Y):
    U = torch.exp(f)
    D = (1 + torch.exp(f))
    yN = Y_N(Y)
    v = -yN * torch.log(U / D) - (1 - yN) * torch.log(1 / D)
    #print("U=%s, D=%s, Y=%s, sum=%s, Y_N=%s" % (U, D, Y, Y_sum, Y_N))

    return torch.sum(v)


class Loss(nn.modules.loss.Module):
    reduction: str

    losses = {
        'OVA': OVALoss,
        'OVA_N': OVALoss_N,
        'PAL': PALLoss,
        'PAL_N': PALLoss_N
    }

    def __init__(self, loss='OVA') -> None:
        super(Loss, self).__init__()

        assert loss in self.losses

        self.loss = self.losses[loss]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss(input, target)


