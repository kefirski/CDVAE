import torch as t
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from utils.functions import *
from torch.autograd import Variable
from torch_modules.other.embedding_lockup import Embedding
from torch_modules.conv_layers.sequential_deconv import SeqDeconv


class HLVAE(nn.Module):
    def __init__(self, params):
        super(HLVAE, self).__init__()

        self.params = params

        self.embedding = Embedding(self.params, '')

        self.encoder = Encoder(self.params)

        self.hidden_to_mu = SeqDeconv(self.params)
        self.hidden_to_logvar = SeqDeconv(self.params)

    def forward(self, drop_prob=0,
                encoder_word_input=None, encoder_character_input=None,
                target_images=None, image_sizes=None,
                decoder_word_input=None,
                z=None):
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param target_images: target image path to estimate image reconstruction loss
        :param image_sizes: sizes of target images
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param z: tensor containing context if sampling is performing
        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 z reconstruction loss
                 adverstal loss result
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.embedding.word_embed.weight.is_cuda

        is_train = fold(lambda acc, parameter: acc and parameter is not None,
                        [encoder_word_input, encoder_character_input, target_images, image_sizes, decoder_word_input],
                        True) and z is None
        is_sampling = fold(lambda acc, parameter: acc and parameter is None,
                           [encoder_word_input, encoder_character_input, target_images, image_sizes,
                            decoder_word_input],
                           True) and z is not None
        assert is_train or is_sampling, 'Invalid input options'

        if is_train:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''

            assert encoder_word_input.size()[0] == len(image_sizes), \
                'while training each batch should be provided with image size to sample with'

            [batch_size, _] = encoder_word_input.size()

            encoder_input = self.embedding(encoder_word_input, encoder_character_input)
            context = self.encoder(encoder_input)

            mu = [self.hidden_to_mu(context[i].unsqueeze(0), size) for i, size in enumerate(image_sizes)]
            logvar = [self.hidden_to_logvar(context[i].unsqueeze(0), size) for i, size in enumerate(image_sizes)]
            std = [t.exp(0.5 * var) for var in logvar]

            z = [HLVAE.sample_z(mu[i], std[i], use_cuda).sigmoid() for i in range(batch_size)]

            return

    @staticmethod
    def sample_z(mu, std, use_cuda):
        """
        Sample differentiable z ~ N(0, I)
        """

        z = Variable(t.rand(mu.size()))
        if use_cuda:
            z = z.cuda()

        return z * std + mu
