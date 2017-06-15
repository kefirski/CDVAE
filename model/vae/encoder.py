import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch_modules.other.highway import Highway


class Encoder(nn.Module):
    def __init__(self, encoder_size, num_layers, embed_size):
        super(Encoder, self).__init__()

        self.encoder_size = encoder_size
        self.num_layers = num_layers
        self.embed_size = embed_size

        self.rnn = nn.GRU(input_size=self.embed_size,
                          hidden_size=self.encoder_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          bidirectional=True)

        self.highway = Highway(self.encoder_size * 2, 4, F.elu)

    def forward(self, input):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """

        [batch_size, _, _] = input.size()

        ''' 
        Unfold rnn with zero initial state and get its final state from the last layer
        '''
        _, final_state = self.rnn(input)
        final_state = final_state \
            .view(self.num_layers, 2, batch_size, self.encoder_size)
        final_state = final_state[-1]
        final_state = t.cat(final_state, 1)

        return self.highway(final_state)