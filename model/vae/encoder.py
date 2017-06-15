import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch_modules.other.highway import Highway
from torch_modules.ln_gru.layer_norm_GRU import LayerNormGRU


class Encoder(nn.Module):
    def __init__(self, encoder_size, num_layers, embed_size):
        super(Encoder, self).__init__()

        self.encoder_size = encoder_size
        self.num_layers = num_layers
        self.embed_size = embed_size

        self.forward_rnn = nn.ModuleList(
            [LayerNormGRU(self.embed_size if i == 0 else self.encoder_size * 2, self.encoder_size)
             for i in range(self.num_layers)])
        self.reverse_rnn = nn.ModuleList(
            [LayerNormGRU(self.embed_size if i == 0 else self.encoder_size * 2, self.encoder_size)
             for i in range(self.num_layers)])

        self.highway = Highway(self.encoder_size * 2, 4, F.elu)

    def forward(self, input):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """

        ''' 
        Unfold rnn with zero initial state and get its final state from the last layer
        '''
        for layer in range(self.num_layers):
            forward_input, _ = self.forward_rnn[layer](input)
            reverse_input, _ = self.reverse_rnn[layer](input)
            input = t.cat([forward_input, reverse_input], 2)

        final_state = input[:, -1]

        return self.highway(final_state)
