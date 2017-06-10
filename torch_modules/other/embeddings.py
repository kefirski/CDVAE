import os
import numpy as np
import torch as t
import torch.nn as nn
from torch.nn import Parameter


class EmbeddingLockup(nn.Module):
    def __init__(self, vocab_size, embed_size, lang: str, path_prefix='../../'):
        super(EmbeddingLockup, self).__init__()

        """
        Loads embedings for language and provides access to it
        """

        assert lang in ['ru', 'en'], 'Invalid lang value. Must be in ["ru", "en"]'

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        word_embeddings_path = path_prefix + 'data/preprocessed_data/word_embeddings_{}.npy'.format(lang)
        assert os.path.exists(word_embeddings_path), 'Word embeddings not found'

        embeddings = np.load(word_embeddings_path)

        self.embeddings = nn.Embedding(self.vocab_size, self.embed_size)
        self.embeddings.weight = Parameter(t.from_numpy(embeddings).float(), requires_grad=False)

    def forward(self, input):
        """
        :param input: [batch_size, seq_len] tensor of Long type
        :return: input embedding with shape of [batch_size, seq_len, word_embed_size]
        """

        return self.embeddings(input)

