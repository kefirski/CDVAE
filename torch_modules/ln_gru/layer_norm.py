import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class LayerNormalization(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.hidden_size = hidden_size
        self.gamma = nn.Parameter(t.ones(1, hidden_size))
        self.betta = nn.Parameter(t.zeros(1, hidden_size))

    def forward(self, z):
        mu = torch.mean(z)
        sigma = torch.std(z)

        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.gamma.expand_as(z) + self.betta.expand_as(z)

        return ln_out
