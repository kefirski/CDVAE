from .functions import *


class Parameters:
    def __init__(self, vocab_size):

        self.vocab_size = vocab_size

        self.embed_size = 15

        self.encoder_size = 8
        self.encoder_num_layers = 3

        self.latent_variable_size = 8

        self.decoder_size = 16
        self.decoder_num_layers = 3
