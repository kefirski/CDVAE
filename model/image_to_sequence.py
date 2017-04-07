import torch as t
import torch.nn as nn
import torch.nn.functional as F
from scipy import misc
from torch.autograd import Variable
from model.encoders.image_encoder import ImageEncoder
from model.decoders.text_decoder import TextDecoder
from utils.functions import *


class ImageToSequence(nn.Module):
    def __init__(self, params, path_prefix):
        super(ImageToSequence, self).__init__()

        self.params = params

        self.image_encoder = ImageEncoder(self.params, path_prefix)
        self.text_decoder = TextDecoder(self.params)

        self.context_to_mu = nn.Linear(512, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(512, self.params.latent_variable_size)

    def forward(self, embeddings,
                drop_prob=0,
                encoder_image_input=None,
                decoder_input=None, initial_state=None,
                z=None):
        """
        :param embeddings: text embedding instance
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param encoder_image_input: array of batch_size length of images paths
        :param decoder_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param initial_state: initial state of decoder rnn in order to perform sampling

        :param z: tensor containing context if sampling is performing
        
        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 decoder final state
                 kld loss estimation
                 mu and logvar
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.context_to_mu.weight.is_cuda

        is_train = z is None

        ''' Get context from encoder and sample z ~ N(mu, std)
        '''

        if is_train:
            assert decoder_input.size()[0] == len(encoder_image_input), \
                'while training each image should be provided with sequence to sample with'

            context = self.image_encoder(encoder_image_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()

            z = sample_z(mu, std, use_cuda)
        else:
            kld = None
            mu = None
            logvar = None

        decoder_input = embeddings.word_embed(decoder_input)
        out, final_state = self.text_decoder(decoder_input, z, drop_prob, initial_state)

        return out, final_state, kld, (mu, logvar)

    def cross_entropy(self, out, target):

        [batch_size, _, _] = out.size()

        out = out.view(-1, self.params.word_vocab_size)
        target = target.view(-1)

        return F.cross_entropy(out, target, size_average=False)/batch_size



