import torch as t
import torch.nn as nn
from scipy import misc
from torch.autograd import Variable
from model.encoders.image_encoder import ImageEncoder
from model.decoders.text_decoder import TextDecoder
from utils.functions import *


class ImageToSequence(nn.Module):
    def __init__(self, params):
        super(ImageToSequence, self).__init__()

        self.params = params

        self.image_encoder = ImageEncoder(self.params)
        self.text_decoder = TextDecoder(self.params)


    def forward(self):
        pass
