from .functions import *


class Parameters:
    def __init__(self, vocab_size):

        self.vocab_size = int(vocab_size)

        self.char_embed_size = 25

        self.text_encoder_size = 12
        self.text_encoder_num_layers = 5
        self.text_decoder_size = 10
        self.text_decoder_num_layers = 5

        self.audio_encoder_size = 6
        self.audio_encoder_num_layers = 6
        self.audio_decoder_size = 6
        self.audio_decoder_num_layers = 6

        self.latent_variable_size = 50


