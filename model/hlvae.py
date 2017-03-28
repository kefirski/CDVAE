import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch_modules.other.embedding_lockup import Embedding
from .encoder import Encoder


class HLVAE(nn.Module):
    def __init__(self, params):

        super(HLVAE, self).__init__()

        self.params = params

        self.embedding = Embedding(self.params, '')

        self.encoder = Encoder(self.params)

    def forward:
        pass
