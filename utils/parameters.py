from .functions import *


class Parameters:
    def __init__(self, vocab_size):

        self.vocab_size = int(vocab_size)

        self.char_embed_size = 19

        self.text_encoder_size = 40
        self.text_encoder_num_layers = 2
        self.text_decoder_size = 40
        self.text_decoder_num_layers = 2

        self.audio_encoder_size = 6
        self.audio_encoder_num_layers = 3
        self.audio_decoder_size = 6
        self.audio_decoder_num_layers = 3

        self.latent_variable_size = 90


