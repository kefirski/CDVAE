from .functions import *


class Parameters:
    def __init__(self, vocab_size_ru, vocab_size_en):
        self.vocab_size_ru = int(vocab_size_ru)
        self.vocab_size_en = int(vocab_size_en)

        self.embed_size = 140

        self.encoder_size = 80
        self.encoder_num_layers = 3

        self.latent_variable_size = 35

        self.decoder_size = 100
        self.decoder_num_layers = 3
