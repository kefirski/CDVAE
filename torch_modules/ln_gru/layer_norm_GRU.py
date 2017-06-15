import torch as t
import torch.nn as nn
from torch.autograd import Variable
from .layer_norm_GRUCell import LayerNormGRUCell


class LayerNormGRU(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = LayerNormGRUCell(self.input_size, self.hidden_size, bias)

    def forward(self, input, state=None):
        """
        :param input: An tensor with shape of [batch_size, seq_len, input_size]
        :param state: An tensor with shape of [batch_size, hidden_size] representing initial state of rnn
        :return: An tensor with shape of [batch_size, seq_len, hidden_size]
                 and final state
        """

        [batch_size, seq_len, _] = input.size()

        result = []

        if state is None:
            state = Variable(t.zeros(batch_size, self.hidden_size))

            if input.is_cuda:
                state = state.cuda()

        for i in range(seq_len):
            state = self.cell(input[:, i], state)
            result += [state.unsqueeze(1)]

        return t.cat(result, 1), state