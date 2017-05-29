import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.autograd import Variable
from torch.nn import Parameter
from .text_decoder import TextDecoder as Decoder
from .text_encoder import TextEncoder as Encoder
from utils.functions import fold


class TextVAE(nn.Module):
    def __init__(self, params):
        super(TextVAE, self).__init__()

        self.params = params

        self.embeddings = nn.Embedding(self.params.vocab_size, self.params.char_embed_size)
        self.embeddings.weight = Parameter(
            t.Tensor(self.params.vocab_size, self.params.char_embed_size).uniform_(-1, 1)
        )

        self.encoder = Encoder(self.params)

        self.context_to_mu = nn.Linear(self.params.text_encoder_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.text_encoder_size * 2, self.params.latent_variable_size)

        self.decoder = Decoder(self.params)

    def forward(self, drop_prob,
                encoder_input=None,
                decoder_input=None,
                z=None, initial_state=None):
        """
        :param drop_prob: probability of an element of decoder input to be dropped out
        :param encoder_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param decoder_input: An tensor with shape of [batch_size, seq_len + 1] of Long type
        :param z: context if sampling is performing
        :param initial_state: initial state of decoder rnn if sampling is performing
        :return: logits of sequence distribution probabilities
                    with shape of [batch_size, seq_len + 1, vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert z is None and fold(lambda acc, par: acc and par is not None, [encoder_input, decoder_input], True) \
            or (z is not None and decoder_input is not None), \
            "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            [batch_size, _] = encoder_input.size()

            encoder_input = self.embeddings(encoder_input)

            context = self.encoder(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if encoder_input.is_cuda:
                z = z.cuda()

            z = z * std + mu

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean()
        else:
            kld = None

        decoder_input = self.embeddings(decoder_input)
        decoder_input = F.dropout(decoder_input, drop_prob, training=z is None)
        out, final_state = self.decoder(decoder_input, z, initial_state)

        return out, final_state, kld

    def sample(self, batch_loader, seq_len, z, use_cuda, path):

        x = batch_loader.text_go_input(1, use_cuda)
        state = None

        result = []

        for i in range(seq_len):
            x, state, _ = self(0., None, x, z, state)
            x = x.squeeze()
            x = F.softmax(x)

            x = x.data.cpu().numpy()
            idx = batch_loader.sample_character_from_distribution(x)
            x = batch_loader.idx_to_char[idx]

            if x == batch_loader.text_stop_token:
                break

            result += [x]

            x = Variable(t.from_numpy(np.array([[idx]]))).long()

            if use_cuda:
                x = x.cuda()

        result = ''.join(result)

        with open("{}.txt".format(path), "w") as f:
            f.write(result)
