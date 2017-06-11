import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .encoder import Encoder
from .decoder import Decoder
from torch_modules.other.embeddings import EmbeddingLockup
from utils.functions import fold


class VAE(nn.Module):
    def __init__(self, encoder_size, encoder_num_layers,
                 decoder_size, decoder_num_layers,
                 latent_size, vocab_size, embed_size,
                 lang: str):
        super(VAE, self).__init__()

        assert lang in ['ru', 'en']

        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lang = lang

        self.embed = EmbeddingLockup(self.vocab_size, self.embed_size, lang, path_prefix='')

        self.encoder = Encoder(encoder_size, encoder_num_layers, self.embed_size)

        self.context_to_mu = nn.Linear(encoder_size * 2, self.latent_size)
        self.context_to_logvar = nn.Linear(encoder_size * 2, self.latent_size)

        self.decoder = Decoder(self.latent_size, decoder_size, decoder_num_layers, self.embed_size, self.vocab_size)

    def forward(self, drop_prob,
                encoder_input=None,
                decoder_input=None,
                z=None, initial_state=None):
        """
        :param drop_prob: probability of an element of decoder input to be dropped out
        :param encoder_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param decoder_input: An tensor with shape of [batch_size, seq_len + 1] of Long type
        :param z: latent variable if sampling is performing
        :param initial_state: initial state of decoder rnn if sampling is performing
        :return: logits of sequence distribution probabilities
                    with shape of [batch_size, seq_len + 1, vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert z is None and fold(lambda acc, par: acc and par is not None, [encoder_input, decoder_input], True) \
               or (z is not None and decoder_input is not None), \
            "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        mu = None
        logvar = None

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)'''
            z, mu, logvar = self.inference(encoder_input)

        out, final_state = self.generate(decoder_input, z, drop_prob, initial_state)

        return out, final_state, mu, logvar

    def inference(self, encoder_input):
        mu, logvar = self.encode(encoder_input)

        z = self.reparametrize(mu, logvar, encoder_input.is_cuda)

        return z, mu, logvar

    def generate(self, decoder_input, z, drop_prob, initial_state):

        decoder_input = self.embed(decoder_input)
        decoder_input = F.dropout(decoder_input, drop_prob, training=z is None)

        return self.decoder(decoder_input, z, initial_state)

    def encode(self, input):
        input = self.embed(input)
        context = self.encoder(input)

        mu = self.context_to_mu(context)
        logvar = self.context_to_logvar(context)

        return mu, logvar

    def reparametrize(self, mu, logvar, use_cuda):

        batch_size = mu.size()[0]

        std = t.exp(0.5 * logvar)

        z = Variable(t.randn([batch_size, self.latent_size]))
        if use_cuda:
            z = z.cuda()

        return z * std + mu

    def sample(self, batch_loader, seq_len, use_cuda, z=None):

        if z is None:
            z = Variable(t.randn(1, self.latent_size))
            if use_cuda:
                z = z.cuda()

        x = batch_loader.go_input(1, self.lang, use_cuda)
        state = None

        result = []

        vocab = [batch_loader.idx_to_word_ru, batch_loader.idx_to_word_en][0 if self.lang == 'ru' else 1]

        for i in range(seq_len):
            x, state, _, _, _ = self(0., None, x, z, state)
            x = x.squeeze()
            x = F.softmax(x)

            x = x.data.cpu().numpy()
            idx = batch_loader.sample_word(x, self.lang)
            x = vocab[idx]

            if x == batch_loader.stop_token:
                break

            result += [x]

            x = Variable(t.from_numpy(np.array([[idx]]))).long()

            if use_cuda:
                x = x.cuda()

        return ' '.join(result)

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
