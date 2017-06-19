from .functions import *


class Parameters:
    def __init__(self, vocab_size):

        self.vocab_size = vocab_size

        self.embed_size = 80

        self.encoder_size = 400
        self.encoder_num_layers = 4

        self.latent_variable_size = 300

        self.decoder_size = 500
        self.decoder_num_layers = 4
