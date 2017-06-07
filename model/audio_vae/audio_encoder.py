import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch_modules.other.highway import Highway


class AudioEncoder(nn.Module):
    def __init__(self, params):
        super(AudioEncoder, self).__init__()

        self.params = params

        self.rnn = nn.GRU(input_size=1,
                          hidden_size=self.params.audio_encoder_size,
                          num_layers=self.params.audio_encoder_num_layers,
                          batch_first=True,
                          bidirectional=False)

        self.highway = Highway(self.params.audio_encoder_size, 4, F.elu)

    def forward(self, input):
        """
        :param input: [batch_size, seq_len, 1] tensor
        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """

        [batch_size, _, _] = input.size()

        ''' 
        Unfold rnn with zero initial state and get its final state from the last layer
        '''
        _, final_state = self.rnn(input)

        final_state = final_state \
            .view(self.params.audio_encoder_num_layers, 1, batch_size, self.params.audio_encoder_size)
        final_state = final_state[-1]
        final_state = t.cat(final_state, 1)

        return self.highway(final_state)
