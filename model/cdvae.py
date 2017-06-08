import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.autograd import Variable
from torch.nn import Parameter
from utils.functions import fold, kld_coef
from .text_vae.text_vae import TextVAE
from .audio_vae.audio_vae import AudioVAE


class CDVAE(nn.Module):
    def __init__(self, params):
        super(CDVAE, self).__init__()

        self.params = params

        self.text_vae = TextVAE(params)
        self.audio_vae = AudioVAE(params)

    def forward(self, text_drop_prob, audio_drop_prob,
                text_encoder_input, audio_encoder_input,
                text_decoder_input, audio_decoder_input,
                text_target, audio_target,
                i):
        """
        :param text_drop_prob: probability of an element of text decoder input to be dropped out
        :param audio_drop_prob: probability of an element of audio decoder input to be dropped out
        :param text_encoder_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param audio_encoder_input: An tensor with shape of [batch_size, seq_len] of Float type
        :param text_decoder_input: An tensor with shape of [batch_size, seq_len + 1] of Long type
        :param audio_decoder_input: An tensor with shape of [batch_size, seq_len + 1] of Float type
        :param text_target: An tensor with shape of [batch_size, seq_len + 1] of Long type
        :param audio_target: An tensor with shape of [batch_size, seq_len + 1] of Float type
        :param i: iteration
        :return: loss estimation for both models
        """

        text_out, _, kld_text, text_mu, text_logvar = \
            self.text_vae(text_drop_prob, text_encoder_input, text_decoder_input)

        text_out = text_out.view(-1, self.params.vocab_size)
        text_target = text_target.view(-1)
        rec_loss_text = F.cross_entropy(text_out, text_target)

        audio_out, _, kld_audio, audio_mu, audio_logvar = \
            self.audio_vae(audio_drop_prob, audio_encoder_input, audio_decoder_input)

        rec_loss_audio = t.pow(audio_out - audio_target, 2).mean()

        cd_latent_loss_text = CDVAE.cd_latent_loss(audio_mu, text_mu, audio_logvar, text_logvar)
        cd_latent_loss_audio = CDVAE.cd_latent_loss(text_mu, audio_mu, text_logvar, audio_logvar)

        '''
        Since ELBO does not contain log(p(x|z)) directly
        but contains quantity that have the same local maximums
        it is necessary to scale this quantity in order to train useful inference model
        '''
        loss_text = 140 * rec_loss_text + kld_coef(i) * kld_text + cd_latent_loss_text
        loss_audio = 2000 * rec_loss_audio + kld_coef(i) * kld_audio + cd_latent_loss_audio

        return (loss_text, rec_loss_text, kld_text, cd_latent_loss_text), \
               (loss_audio, rec_loss_audio, kld_audio, cd_latent_loss_audio)

    @staticmethod
    def cd_latent_loss(mu_1, mu_2, logvar_1, logvar_2):
        return 0.5 * t.sum(logvar_1 - logvar_2 + t.exp(logvar_2) / t.exp(logvar_1) +
                           t.pow(mu_1 - mu_2, 2) / t.exp(logvar_1) - 1).mean()
