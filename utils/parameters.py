from .functions import *


class Parameters:
    def __init__(self, max_word_len, max_seq_len, word_vocab_size, char_vocab_size):
        self.max_word_len = int(max_word_len)
        self.max_seq_len = int(max_seq_len) + 1  # go or end token

        self.word_vocab_size = int(word_vocab_size)
        self.char_vocab_size = int(char_vocab_size)

        self.word_embed_size = 300
        self.char_embed_size = 15

        self.kernels = [(1, 25), (2, 50), (3, 75), (4, 100), (5, 125), (6, 150)]
        self.sum_depth = fold(lambda x, y: x + y, [depth for _, depth in self.kernels], 0)

        self.encoder_rnn_size = 250
        self.encoder_num_layers = 2

        self.hidden_size = 16384
        self.hidden_view = [256, 8, 8]

        # (input_channels, output_channels, kernel_size, (out_h, out_w))
        self.deconv_kernels = [(256, 256, 5, (16, 16)),
                               (256, 92, 5, (32, 32)),
                               (92, 92, 5, (64, 64)),
                               (92, 3, 5, (128, 128))]
        self.deconv_num_layers = len(self.deconv_kernels)
