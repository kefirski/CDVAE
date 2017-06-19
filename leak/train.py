import argparse
import os
import numpy as np
import torch as t
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal
from torch.optim import Adam


def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))


def f_and(x, y):
    return x and y


def f_or(x, y):
    return x or y

class Decoder(nn.Module):
    def __init__(self, latent_size, decoder_size, num_layers, embed_size, vocab_size):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        self.decoder_size = decoder_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        self.rnn = nn.GRU(input_size=self.embed_size + self.latent_size,
                          hidden_size=self.decoder_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.fc = nn.Linear(self.decoder_size, self.vocab_size)

    def forward(self, decoder_input, z, initial_state=None):

        [batch_size, seq_len, _] = decoder_input.size()

        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_input = t.cat([decoder_input, z], 2)

        result, final_state = self.rnn(decoder_input, initial_state)

        result = result.contiguous().view(-1, self.decoder_size)
        result = self.fc(result)
        result = result.view(batch_size, seq_len, self.vocab_size)

        return result, final_state

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

    def forward(self, input):

        [batch_size, _, _] = input.size()
        
        _, final_state = self.rnn(input)
        final_state = final_state \
            .view(self.num_layers, 2, batch_size, self.encoder_size)
        final_state = final_state[-1]

        return t.cat(final_state, 1)

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

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.embed.weight = xavier_normal(self.embed.weight)

        self.encoder = Encoder(encoder_size, encoder_num_layers, self.embed_size)

        self.context_to_mu = nn.Linear(encoder_size * 2, self.latent_size)
        self.context_to_logvar = nn.Linear(encoder_size * 2, self.latent_size)

        self.decoder = Decoder(self.latent_size, decoder_size, decoder_num_layers, self.embed_size, self.vocab_size)

    def forward(self, drop_prob,
                encoder_input=None,
                decoder_input=None,
                z=None, initial_state=None):

        mu = None
        logvar = None

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)'''
            z, mu, logvar = self.inference(encoder_input)

        out, final_state = self.generate(decoder_input, z, drop_prob, initial_state)

        return out, final_state, mu, logvar

    def inference(self, encoder_input):
        encoder_input = self.embed(encoder_input)
        
        context = self.encoder(encoder_input)

        mu = self.context_to_mu(context)
        logvar = self.context_to_logvar(context)

        z = self.reparametrize(mu, logvar, encoder_input.is_cuda)

        return z, mu, logvar

    def generate(self, decoder_input, z, drop_prob, initial_state):
        decoder_input = self.embed(decoder_input)
        decoder_input = F.dropout(decoder_input, drop_prob, training=z is None)

        return self.decoder(decoder_input, z, initial_state)

    def reparametrize(self, mu, logvar, use_cuda):

        batch_size = mu.size()[0]

        std = t.exp(0.5 * logvar)

        z = Variable(t.randn([batch_size, self.latent_size]))
        if use_cuda:
            z = z.cuda()

        return z * std + mu

    def sample(self, seq_len, use_cuda, z=None):

        if z is None:
            z = Variable(t.randn(1, self.latent_size))
            if use_cuda:
                z = z.cuda()

        x = Variable(t.LongTensor([[1]]))
        if use_cuda:
            x = x.cuda()
        state = None

        result = []

        for i in range(seq_len):
            x, state, _, _ = self(0., None, x, z, state)
            x = x.squeeze()
            x = F.softmax(x)

            x = x.data.cpu().numpy()
            idx = np.random.choice(len(x), p=x.ravel())
            x = 'q'

            result += [x]

            x = Variable(t.from_numpy(np.array([[idx]]))).long()

            if use_cuda:
                x = x.cuda()

        return ''.join(result)


class CDVAE(nn.Module):

    def __init__(self, encoder_size, encoder_num_layers,
				decoder_size, decoder_num_layers,
				latent_variable_size, vocab_size, embed_size):
        super(CDVAE, self).__init__()


        self.vae_ru = VAE(encoder_size, encoder_num_layers,
                          decoder_size, decoder_num_layers,
                          latent_variable_size, vocab_size, embed_size, 'ru')

        self.vae_en = VAE(encoder_size, encoder_num_layers,
                          decoder_size, decoder_num_layers,
                          latent_variable_size, vocab_size, embed_size, 'en')

    def forward(self, drop_prob,
                encoder_input_ru, encoder_input_en,
                decoder_input_ru, decoder_input_en,
                target_ru, target_en,
                i):

        ce_ru, kld_ru, mu_ru, logvar_ru = self.loss(encoder_input_ru, decoder_input_ru, target_ru, drop_prob, 'ru')
        ce_en, kld_en, mu_en, logvar_en = self.loss(encoder_input_en, decoder_input_en, target_en, drop_prob, 'en')

        cd_kld_ru = CDVAE.cd_latent_loss(mu_ru, logvar_ru, mu_en, logvar_en)
        cd_kld_en = CDVAE.cd_latent_loss(mu_en, logvar_en, mu_ru, logvar_ru)

        '''
        Since ELBO does not contain log(p(x|z)) directly
        but contains quantity that have the same local maximums
        it is necessary to scale this quantity in order to train useful inference model
        '''
        loss_ru = 850 * ce_ru + kld_ru + cd_kld_ru
        loss_en = 850 * ce_en + kld_en + cd_kld_en

        return (loss_ru, ce_ru, kld_ru, cd_kld_ru), \
               (loss_en, ce_en, kld_en, cd_kld_en)

    def loss(self, encoder_input, decoder_input, decoder_target, drop_prob: float, lang: str):

        model = [self.vae_ru, self.vae_en][0 if lang == 'ru' else 1]

        out, _, mu, logvar = model(drop_prob, encoder_input, decoder_input)
        
        vs = out.size()[2]
        out = out.view(-1, vs)
        decoder_target = decoder_target.view(-1)

        cross_entropy = F.cross_entropy(out, decoder_target)
        kld = CDVAE.latent_loss(mu, logvar)

        return cross_entropy, kld, mu, logvar

    def translate(self, encoder_input, from_to: list):

        model_from = [self.vae_ru, self.vae_en][0 if from_to[0] == 'ru' else 1]
        z, _, _ = model_from.inference(encoder_input)

        model_to = [self.vae_ru, self.vae_en][0 if from_to[1] == 'ru' else 1]

        return model_to.sample(encoder_input.size()[1], encoder_input.is_cuda, z)

    @staticmethod
    def cd_latent_loss(mu_1, logvar_1, mu_2, logvar_2):
        return 0.5 * t.sum(logvar_2 - logvar_1 + t.exp(logvar_1) / (t.exp(logvar_2) + 1e-8) +
                           t.pow(mu_1 - mu_2, 2) / (t.exp(logvar_2) + 1e-8) - 1).mean()

    @staticmethod
    def latent_loss(mu, logvar):
        return (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CDVAE')
    parser.add_argument('--num-iterations', type=int, default=450000, metavar='NI',
                        help='num iterations (default: 450000)')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS',
                        help='batch size (default: 30)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--dropout', type=float, default=0.12, metavar='TDR',
                        help='dropout (default: 0.12)')
    args = parser.parse_args()

    cdvae = CDVAE(encoder_size=300, encoder_num_layers=3,
				decoder_size=400, decoder_num_layers=3,
				latent_variable_size=200, vocab_size=100, embed_size=40)
    if args.use_cuda:
        cdvae = cdvae.cuda()

    optimizer_ru = Adam(cdvae.vae_ru.parameters(), args.learning_rate, eps=1e-4)
    optimizer_en = Adam(cdvae.vae_en.parameters(), args.learning_rate, eps=1e-4)

    for iteration in range(args.num_iterations):

        print(iteration)

        input = (Variable(t.rand(args.batch_size, 100)).long() for _ in range(6))
        input, dec_input_ru, dec_target_ru, input_en, dec_input_en, dec_target_en = input
        if args.use_cuda:
            input, dec_input_ru, dec_target_ru, input_en, dec_input_en, dec_target_en = (var.cuda() for var in input)
        
        

        '''losses from cdvae is tuples of ru and en losses respectively'''
        loss_ru, loss_en = cdvae(args.dropout,
                                 input, input_en,
                                 dec_input_ru, dec_input_en,
                                 dec_target_ru, dec_target_en,
                                 iteration)

        optimizer_ru.zero_grad()
        loss_ru[0].backward(retain_variables=True)
        optimizer_ru.step()

        optimizer_en.zero_grad()
        loss_en[0].backward()
        optimizer_en.step()

        if iteration % 20 == 0:
            input = Variable(t.rand(1, 100)).long()
            if args.use_cuda:
                input = input.cuda()
            cdvae.translate(input, ['ru', 'en'])

            input = Variable(t.rand(1, 300)).long()
            cdvae.translate(input, ['ru', 'ru'])
