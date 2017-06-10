import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.functions import fold, kld_coef
from .vae.vae import VAE


class CDVAE(nn.Module):
    def __init__(self, params):
        super(CDVAE, self).__init__()

        self.params = params

        self.vae_ru = VAE(params.encoder_size, params.encoder_num_layers, params.decoder_size,
                          params.decoder_num_layers,
                          params.latent_variable_size, params.vocab_size_ru, params.embed_size, 'ru')

        self.vae_en = VAE(params.encoder_size, params.encoder_num_layers, params.decoder_size,
                          params.decoder_num_layers,
                          params.latent_variable_size, params.vocab_size_en, params.embed_size, 'en')

    def forward(self, drop_prob,
                encoder_input_ru, encoder_input_en,
                decoder_input_ru, decoder_input_en,
                target_ru, target_en,
                i):
        """
        :param drop_prob: probability of an element of decoder input to be dropped out
        :param encoder_input_ru: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_input_en: An tensor with shape of [batch_size, seq_len] of Long type
        :param decoder_input_ru: An tensor with shape of [batch_size, seq_len + 1] of Long type
        :param decoder_input_en: An tensor with shape of [batch_size, seq_len + 1] of Long type
        :param target_ru: An tensor with shape of [batch_size, seq_len + 1] of Long type
        :param target_en: An tensor with shape of [batch_size, seq_len + 1] of Long type
        :param i: iteration
        :return: loss estimation for both models
        """

        out_ru, _, kld_ru, mu_ru, logvar_ru = \
            self.vae_ru(drop_prob, encoder_input_ru, decoder_input_ru)

        out_ru = out_ru.view(-1, self.params.vocab_size_ru)
        target_ru = target_ru.view(-1)
        rec_loss_ru = F.cross_entropy(out_ru, target_ru)

        out_en, _, kld_en, mu_en, logvar_en = \
            self.vae_en(drop_prob, encoder_input_en, decoder_input_en)

        out_en = out_en.view(-1, self.params.vocab_size_en)
        target_en = target_en.view(-1)
        rec_loss_en = F.cross_entropy(out_en, target_en)

        cd_latent_loss_ru = CDVAE.cd_latent_loss(mu_en, mu_ru, logvar_en, logvar_ru)
        cd_latent_loss_en = CDVAE.cd_latent_loss(mu_ru, mu_en, logvar_ru, logvar_en)

        '''
        Since ELBO does not contain log(p(x|z)) directly
        but contains quantity that have the same local maximums
        it is necessary to scale this quantity in order to train useful inference model
        '''
        loss_ru = 83 * rec_loss_ru + kld_coef(i) * kld_ru + cd_latent_loss_ru
        loss_en = 83 * rec_loss_en + kld_coef(i) * kld_en + cd_latent_loss_en

        return (loss_ru, rec_loss_ru, kld_ru, cd_latent_loss_ru), \
               (loss_en, rec_loss_en, kld_en, cd_latent_loss_en)

    @staticmethod
    def cd_latent_loss(mu_1, mu_2, logvar_1, logvar_2):
        return 0.5 * t.sum(logvar_1 - logvar_2 + t.exp(logvar_2) / t.exp(logvar_1) +
                           t.pow(mu_1 - mu_2, 2) / t.exp(logvar_1) - 1).mean()
