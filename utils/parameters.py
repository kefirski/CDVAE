from .functions import *


class Parameters:
    def __init__(self, vocab_size):

        self.vocab_size = vocab_size

        self.embed_size = 35

        self.encoder_size = 25
        self.encoder_num_layers = 5

        self.latent_variable_size = 35

        self.decoder_size = 27
        self.decoder_num_layers = 6
