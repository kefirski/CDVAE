import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch_modules.other.highway import Highway


class AudioDecoder(nn.Module):
    def __init__(self, params):
        super(AudioDecoder, self).__init__()

        self.params = params

        self.rnn = nn.GRU(input_size=self.params.latent_variable_size + 1,
                          hidden_size=self.params.audio_decoder_size,
                          num_layers=self.params.audio_decoder_num_layers,
                          batch_first=True)

        self.highway = Highway(self.params.audio_decoder_size, 4, F.elu)
        self.fc = nn.Linear(self.params.audio_decoder_size, 1)

    def forward(self, decoder_input, z, initial_state=None):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, 1]
        :param z: latent variable with shape of [batch_size, latent_variable_size]
        :param initial_state: initial state of generator rnn
        :return: mean of audio distribution probabilities
                    with shape of [batch_size, seq_len, 1]
                 final rnn state with shape of [num_layers, batch_size, audio_decoder_size]
        """

        [batch_size, seq_len, _] = decoder_input.size()

        '''decoder rnn is conditioned on context via additional bias = W_cond * z applied to every input token'''
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_input = t.cat([decoder_input, z], 2)

        result, final_state = self.rnn(decoder_input, initial_state)

        result = result.contiguous().view(-1, self.params.audio_decoder_size)
        result = self.highway(result)
        result = F.tanh(self.fc(result))
        result = result.view(batch_size, seq_len, 1)

        return result, final_state
