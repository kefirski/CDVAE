import torch.nn as nn
from torch_modules.other.embedding_lockup import EmbeddingLockup
from model.sequence_to_image import SequenceToImage
from model.image_to_sequence import ImageToSequence
from model.disсriminator import Disсriminator


class CDVAE(nn.Module):
    def __init__(self, params, path_prefix):
        super(CDVAE, self).__init__()

        self.params = params
        self.path_prefix = path_prefix

        self.embeddings = EmbeddingLockup(self.params, self.path_prefix)

        """
        takes batch size of sequences and sample appropriate images
        discriminator network uses to make images more realistic
        """
        self.seq_to_image = SequenceToImage(params)
        self.discriminator = Disсriminator(params, self.path_prefix)

        """
        takes array of images of batch size length to emit batch size of sequences
        model uses decoder context input in pair with latent representation
        """
        self.image_to_seq = ImageToSequence(params, self.path_prefix)

    def forward(self, drop_prob=0,
                encoder_word_input=None, encoder_character_input=None,
                target_images=None, target_image_sizes=None,
                decoder_word_input=None):
        """
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout 
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param target_images: array of batch_size length of images paths
        :param target_image_sizes: sizes of target images 
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        
        This method is necessary to forward propagate of both seq_to_image and image_to_seq models
        In order to sample data from decoders of these models use :sample_image: and :sample_seq: methods
        """

        seq_to_image_result = self.seq_to_image(self.embeddings,
                                                drop_prob=drop_prob,
                                                encoder_word_input=encoder_word_input,
                                                encoder_character_input=encoder_character_input,
                                                target_sizes=target_image_sizes)

        image_to_seq_result = self.image_to_seq(self.embeddings,
                                                drop_prob=drop_prob,
                                                encoder_image_input=target_images,
                                                decoder_input=decoder_word_input)

        return seq_to_image_result, image_to_seq_result

    def seq_to_image_parameters(self):
        return [p for p in self.seq_to_image.parameters() if p.requires_grad]

    def image_to_seq_parameters(self):
        return [p for p in self.image_to_seq.parameters() if p.requires_grad]

    def discriminator_parameters(self):
        return [p for p in self.discriminator.parameters() if p.requires_grad]

    def trainer(self, s2i_optimizer, i2s_optimizer, discriminator_optimizer, batch_loader):
        def train(batch_size, use_cuda, drop_prob):

            word_level_encoder_input, character_level_encoder_input, target_images, \
                real_images, target_images_sizes, decoder_text_input, decoder_text_target = \
                batch_loader.next_batch(batch_size, 'train')

            [word_level_encoder_input, character_level_encoder_input, decoder_text_input, decoder_text_target] = \
                [Variable(t.from_numpy(var)) for var in
                 [word_level_encoder_input, character_level_encoder_input, decoder_text_input, decoder_text_target]]

            if use_cuda:
                [word_level_encoder_input, character_level_encoder_input, decoder_text_input, decoder_text_target] = \
                [var.cuda() for var in
                 [word_level_encoder_input, character_level_encoder_input, decoder_text_input, decoder_text_target]]

