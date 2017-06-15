import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch_modules.other.highway import Highway
from torch_modules.ln_gru.layer_norm_GRU import LayerNormGRU


class Decoder(nn.Module):
    def __init__(self, latent_size, decoder_size, num_layers, embed_size, vocab_size):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        self.decoder_size = decoder_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        self.rnn = nn.ModuleList(
            [LayerNormGRU(self.embed_size + self.latent_size if i == 0 else self.decoder_size, self.decoder_size) for i
             in range(self.num_layers)])

        self.highway = Highway(self.decoder_size, 3, F.elu)
        self.fc = nn.Linear(self.decoder_size, self.vocab_size)

    def forward(self, decoder_input, z, initial_state):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: latent variable with shape of [batch_size, latent_variable_size]
        :param initial_state: initial state of generator rnn
        :return: unnormalized logits of sentense characters distribution probabilities
                    with shape of [batch_size, seq_len, embed_size]
                 final rnn state with shape of [num_layers, batch_size, text_decoder_size]
        """

        '''
        Takes decoder input with latent variable 
        and predicts distribution of probabilities over vords in vocabulary.
         
        Decoder rnn is conditioned on context via additional bias = W_cond * z
        applied to every token in input
        '''

        [batch_size, seq_len, _] = decoder_input.size()

        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_input = t.cat([decoder_input, z], 2)

        if initial_state is None:
            initial_state = [None] * self.num_layers

        final_state = [None] * self.num_layers

        for layer in range(self.num_layers):
            decoder_input, final_state_layer = self.rnn[layer](decoder_input, initial_state[layer])
            final_state[layer] = final_state_layer

        result = decoder_input.contiguous().view(-1, self.decoder_size)
        result = self.highway(result)
        result = self.fc(result)
        result = result.view(batch_size, seq_len, self.vocab_size)

        return result, final_state
