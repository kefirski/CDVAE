import torch.nn as nn
from selfModules.embedding import Embedding
from model.sequence_to_image import SequenceToImage
from model.image_to_sequence import ImageToSequence


class CDVAE(nn.Module):
    def __init__(self, params, path_prefix):
        super(CDVAE, self).__init__()

        self.params = params
        self.path_prefix = path_prefix

        embedding = Embedding(self.params, self.path_prefix)

        self.seq_to_image = SequenceToImage(params)
        self.image_to_seq = ImageToSequence(params)

    def forward(self, drop_prob=0,
                encoder_word_input=None, encoder_character_input=None,
                target_images=None, target_image_sizes=None,
                real_images=None,
                decoder_word_input=None,
                z=None):
        pass

