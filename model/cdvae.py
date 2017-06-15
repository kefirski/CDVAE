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

        self.vae_ru = VAE(params.encoder_size, params.encoder_num_layers,
                          params.decoder_size, params.decoder_num_layers,
                          params.latent_variable_size, params.vocab_size['ru'], params.embed_size, 'ru')

        self.vae_en = VAE(params.encoder_size, params.encoder_num_layers,
                          params.decoder_size, params.decoder_num_layers,
                          params.latent_variable_size, params.vocab_size['en'], params.embed_size, 'en')

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

        ce_ru, kld_ru, mu_ru, logvar_ru = self.loss(encoder_input_ru, decoder_input_ru, target_ru, drop_prob, 'ru')
        ce_en, kld_en, mu_en, logvar_en = self.loss(encoder_input_en, decoder_input_en, target_en, drop_prob, 'en')

        cd_kld_ru = CDVAE.cd_latent_loss(mu_ru, logvar_ru, mu_en, logvar_en)
        cd_kld_en = CDVAE.cd_latent_loss(mu_en, logvar_en, mu_ru, logvar_ru)

        '''
        Since ELBO does not contain log(p(x|z)) directly
        but contains quantity that have the same local maximums
        it is necessary to scale this quantity in order to train useful inference model
        '''
        loss_ru = 500 * ce_ru + kld_coef(i) * (kld_ru + cd_kld_ru)
        loss_en = 500 * ce_en + kld_coef(i) * (kld_en + cd_kld_en)

        return (loss_ru, ce_ru, kld_ru, cd_kld_ru), \
               (loss_en, ce_en, kld_en, cd_kld_en)

    def loss(self, encoder_input, decoder_input, decoder_target, drop_prob: float, lang: str):
        """
        :param encoder_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param decoder_input: An tensor with shape of [batch_size, seq_len + 1] of Long type
        :param decoder_target: An tensor with shape of [batch_size, seq_len + 1] of Long type
        :param drop_prob: probability of an element of decoder input to be dropped out
        :param lang: language to choose model from
        :return: ELBO parts, mu and logvar of inference reparametrization
        """

        model = [self.vae_ru, self.vae_en][0 if lang == 'ru' else 1]

        out, _, mu, logvar = model(drop_prob, encoder_input, decoder_input)

        out = out.view(-1, self.params.vocab_size[lang])
        decoder_target = decoder_target.view(-1)

        cross_entropy = F.cross_entropy(out, decoder_target)
        kld = CDVAE.latent_loss(mu, logvar)

        return cross_entropy, kld, mu, logvar

    def translate(self, encoder_input_from, decoder_input_to, to: str):
        """
        :param encoder_input_from: An tensor with shape of [batch_size, seq_len] of Long type
        :param decoder_input_to: An tensor with shape of [batch_size, seq_len + 1] of Long type
        :param to: language to choose model from
        :return: a numpy array of generated data
        """

        '''
        Performs inference from one model 
        and generate data with condition to latent variable from another
        '''

        model_from = [self.vae_ru, self.vae_en][0 if to == 'en' else 1]
        z, _, _ = model_from.inference(encoder_input_from)

        model_to = [self.vae_ru, self.vae_en][0 if to == 'ru' else 1]
        translation, _ = model_to.generate(decoder_input_to, z, 0.5, None)

        [batch_size, seq_len, vocab_size] = translation.size()

        translation = translation.view(-1, vocab_size)
        translation = F.softmax(translation)
        translation = translation.view(batch_size, seq_len, vocab_size)

        return translation.data.cpu().numpy()

    @staticmethod
    def cd_latent_loss(mu_1, logvar_1, mu_2, logvar_2):
        return 0.5 * t.sum(logvar_2 - logvar_1 + t.exp(logvar_1) / t.exp(logvar_2) +
                           t.pow(mu_1 - mu_2, 2) / t.exp(logvar_2) - 1).mean()

    @staticmethod
    def latent_loss(mu, logvar):
        return (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean()
