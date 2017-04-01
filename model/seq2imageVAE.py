import torch as t
import torch.nn as nn
import numpy as np
from scipy import misc
import torch.nn.functional as F
from .text_encoder import TextEncoder
from utils.functions import *
from torch.autograd import Variable
from torch_modules.other.embedding_lockup import Embedding
from torch_modules.conv_layers.sequential_deconv import SeqDeconv


class SeqToImage(nn.Module):
    def __init__(self, params):
        super(SeqToImage, self).__init__()

        self.params = params

        self.embedding = Embedding(self.params, '')

        self.encoder = TextEncoder(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        self.hidden_to_image = nn.Linear(self.params.latent_variable_size, self.params.hidden_size)

        self.unroll_image = SeqDeconv(self.params)

        [self.input_channels, self.h, self.w] = self.params.hidden_view

    def forward(self, drop_prob=0,
                encoder_word_input=None, encoder_character_input=None,
                target_images=None, target_image_sizes=None,
                real_images=None,
                decoder_word_input=None,
                z=None):
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param target_images: target images path to estimate image reconstruction loss
        :param target_image_sizes: sizes of target images
        :param real_images: real images to estimate adverstal loss
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param z: tensor containing context if sampling is performing
        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 bce between latent representation and target representation
                 adverstal loss result
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.embedding.word_embed.weight.is_cuda

        is_train = fold(lambda acc, parameter: acc and parameter is not None,
                        [encoder_word_input, encoder_character_input, target_images, target_image_sizes, decoder_word_input],
                        True) and z is None
        is_sampling = fold(lambda acc, parameter: acc and parameter is None,
                           [encoder_word_input, encoder_character_input, target_images, target_image_sizes,
                            decoder_word_input],
                           True) and z is not None
        assert is_train or is_sampling, 'Invalid input options'

        ''' Get context from encoder and sample z ~ N(mu, std)
        '''

        if is_train:
            assert encoder_word_input.size()[0] == len(target_image_sizes), \
                'while training each batch should be provided with image size to sample with'

            [batch_size, _] = encoder_word_input.size()

            encoder_input = self.embedding(encoder_word_input, encoder_character_input)
            context = self.encoder(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = SeqToImage.sample_z(mu, std, use_cuda)

            z = self.hidden_to_image(z)
            z = [z[i].unsqueeze(0) for i in range(batch_size)]

            z = [self.unroll_image(z[i].view(-1, self.input_channels, self.h, self.w), target_image_sizes[i])
                 for i in range(batch_size)]

            mse = t.cat([SeqToImage.mse(z[i], target_images[i]) for i in range(batch_size)]).mean()


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
    def mse(z, image_path):
        """
        :param z: tensor with shape of [1, 3, height, width]
        :return: MSE between latent representation z and target representation
        """

        image = misc.imread(image_path)/255
        image = (Variable(t.from_numpy(image))).float().transpose(2, 0).contiguous()
        if z.is_cuda:
            image = image.cuda()

        z = z.squeeze(0).contiguous().view(-1)
        image = image.view(-1)

        return t.pow(z - image, 2).mean()

