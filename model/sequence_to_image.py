import torch as t
import torch.nn as nn
from scipy import misc
from torch.autograd import Variable

from model.decoders.image_decoder import ImageDecoder
from model.encoders.text_encoder import TextEncoder
from torch_modules.other.embedding_lockup import Embedding
from utils.functions import *


class SequenceToImage(nn.Module):
    def __init__(self, params):
        super(SequenceToImage, self).__init__()

        self.params = params

        self.embedding = Embedding(self.params, '')

        self.encoder = TextEncoder(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        self.hidden_to_image_size = nn.Linear(self.params.latent_variable_size, self.params.hidden_size)

        self.unroll_image = ImageDecoder(self.params)

        [self.input_channels, self.h, self.w] = self.params.hidden_view

    def forward(self, drop_prob=0,
                encoder_word_input=None, encoder_character_input=None,
                target_image_sizes=None,
                decoder_word_input=None,
                z=None):
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param target_image_sizes: sizes of target images
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param z: tensor containing context if sampling is performing
        :return: An array of result images with shape of [3, height_i, width_i]
                 kld loss estimation 
                 mu and logvar
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.embedding.word_embed.weight.is_cuda

        is_train = z is None

        ''' Get context from encoder and sample z ~ N(mu, std)
        '''

        if is_train:
            assert encoder_word_input.size()[0] == len(target_image_sizes), \
                'while training each batch should be provided with image size to sample with'

            encoder_input = self.embedding(encoder_word_input, encoder_character_input)
            context = self.encoder(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()

            z = SequenceToImage.sample_z(mu, std, use_cuda)

        else:
            kld = None
            mu = None
            logvar = None

        z = self.hidden_to_image_size(z)
        z = [var.view(self.input_channels, self.h, self.w) for var in z]

        z = [self.unroll_image(var, target_image_sizes[i]).sigmoid()
             for i, var in enumerate(z)]

        return z, kld, (mu, logvar)

    @staticmethod
    def sample_z(mu, std, use_cuda):
        """
        :return: differentiable z ~ N(mu, std)
        """

        z = Variable(t.rand(mu.size()))
        if use_cuda:
            z = z.cuda()

        return z * std + mu

    @staticmethod
    def mse(z, image_paths):
        """
        :param z: An array of tensos with shape of [3, height, width]
        :return: MSE between latent representation z and target representation
        """

        mse = []

        for i, var in enumerate(z):
            image = misc.imread(image_paths[i])/255
            image = (Variable(t.from_numpy(image))).float().transpose(2, 0).contiguous()
            if var.is_cuda:
                image = image.cuda()

            var = var.contiguous().view(-1)
            image = image.view(-1)

            mse += [t.pow(var - image, 2).mean()]

        return t.cat(mse)

