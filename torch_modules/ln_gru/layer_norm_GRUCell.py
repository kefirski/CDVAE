import math
import torch as t
from torch.nn.module import Module
from torch.nn import Parameter
from .layer_norm import LayerNormalization


class LayerNormGRUCell(Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(t.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(t.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(t.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(t.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.ln_ih = LayerNormalization(3 * hidden_size)
        self.ln_hh = LayerNormalization(3 * hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):

        gi = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih))
        gh = self.ln_hh(F.linear(hidden, self.weight_hh, self.bias_hh))
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy
