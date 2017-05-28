import torch as t
import torch.nn as nn
import torch.nn.functional as F
from selfModules.highway import Highway


class TextDecoder(nn.Module):
    def __init__(self, params):
        super(TextDecoder, self).__init__()

        self.params = params

        self.rnn = nn.GRU(input_size=self.params.char_embed_size + self.params.latent_variable_size,
                          hidden_size=self.params.text_decoder_size,
                          num_layers=self.params.text_decoder_num_layers,
                          batch_first=True)

        self.highway = Highway(self.params.text_decoder_size, 3, F.elu)
        self.fc = nn.Linear(self.params.text_decoder_size, self.params.char_embed_size)

    def forward(self, decoder_input, z, initial_state=None):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: latent variable with shape of [batch_size, latent_variable_size]
        :param initial_state: initial state of generator rnn
        :return: unnormalized logits of sentense characters distribution probabilities
                    with shape of [batch_size, seq_len, embed_size]
                 final rnn state with shape of [num_layers, batch_size, text_decoder_size]
        """

        [batch_size, seq_len, _] = decoder_input.size()

        '''decoder rnn is conditioned on context via additional bias = W_cond * z applied to every input token'''
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_input = t.cat([decoder_input, z], 2)

        result, final_state = self.rnn(decoder_input, initial_state)

        result = result.contiguous().view(-1, self.params.text_decoder_size)
        result = self.highway(result)
        result = self.fc(result)
        result = result.view(batch_size, seq_len, self.params.char_embed_size)

        return result, final_state
