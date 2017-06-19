from .functions import *


class Parameters:
    def __init__(self, vocab_size):

        self.vocab_size = vocab_size

        self.embed_size = 80

        self.encoder_size = 300
        self.encoder_num_layers = 2

        self.latent_variable_size = 300

        self.decoder_size = 350
        self.decoder_num_layers = 3