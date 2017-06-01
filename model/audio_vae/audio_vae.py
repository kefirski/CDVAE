import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.autograd import Variable
from .audio_decoder import AudioDecoder as Decoder
from .audio_encoder import AudioEncoder as Encoder
from utils.functions import fold


class AudioVAE(nn.Module):
    def __init__(self, params):
        super(AudioVAE, self).__init__()

        self.params = params

        self.encoder = Encoder(self.params)

        self.context_to_mu = nn.Linear(self.params.audio_encoder_size, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.audio_encoder_size, self.params.latent_variable_size)

        self.decoder = Decoder(self.params)

    def forward(self, drop_prob,
                encoder_input=None,
                decoder_input=None,
                z=None, initial_state=None):
        """
        :param drop_prob: probability of an element of decoder input to be dropped out
        :param encoder_input: An tensor with shape of [batch_size, seq_len] of Float type
        :param decoder_input: An tensor with shape of [batch_size, seq_len + 1] of Float type
        :param z: context if sampling is performing
        :param initial_state: initial state of decoder rnn if sampling is performing
        :return: mu of N(mu, var) of sequence distribution probabilities
                    with shape of [batch_size, seq_len + 1]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """
        assert z is None and fold(lambda acc, par: acc and par is not None, [encoder_input, decoder_input], True) \
               or (z is not None and decoder_input is not None), \
            "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        mu = None
        logvar = None

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            [batch_size, _] = encoder_input.size()

            mu, logvar = self.encode(encoder_input)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if encoder_input.is_cuda:
                z = z.cuda()

            z = z * std + mu

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean()
        else:
            kld = None

        decoder_input = decoder_input.unsqueeze(2)
        decoder_input = F.dropout(decoder_input, drop_prob, training=z is None)
        out, final_state = self.decoder(decoder_input, z, initial_state)

        return out.squeeze(2), final_state, kld, mu, logvar

    def encode(self, input):
        input = input.unsqueeze(2)
        context = self.encoder(input)

        mu = self.context_to_mu(context)
        logvar = self.context_to_logvar(context)

        return mu, logvar

    def sample(self, batch_loader, seq_len, use_cuda, z=None):

        if z is None:
            z = Variable(t.randn(1, self.params.latent_variable_size))
            if use_cuda:
                z = z.cuda()

        x = batch_loader.audio_go_input(1, use_cuda)
        state = None

        result = []

        for i in range(seq_len):
            x, state, _, _, _ = self(0., None, x, z, state)

            x = x.squeeze()
            x = x.data.cpu().numpy()[0]

            if x == batch_loader.audio_stop_token:
                break

            result += [x]

            x = Variable(t.from_numpy(np.array([[x]]))).float()

            if use_cuda:
                x = x.cuda()

        return np.array(result)
