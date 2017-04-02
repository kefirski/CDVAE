import math
import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from scipy import misc
from torch.nn import Parameter
from torch.autograd import Variable
from torch_modules.other.expand_with_zeros import expand_with_zeroes
from torch_modules.other.highway import Highway


class ImageEncoder(nn.Module):
    def __init__(self, params):
        super(ImageEncoder, self).__init__()

        self.params = params

        self.conv_weights = nn.ParameterList([Parameter(t.Tensor(out_c, in_c, kernel_size, kernel_size))
                                              for out_c, in_c, kernel_size in self.params.encoder_kernels])
        self.conv_biases = nn.ParameterList([Parameter(t.Tensor(out_c)) for out_c, _, _ in self.params.encoder_kernels])

        for weight in self.conv_weights:
            init.xavier_uniform(weight, gain=math.sqrt(2.0))

        self.out_size = image_encoder_out_size

        self.hw = Highway(self.out_size, 2, F.relu)

    def forward(self):
        pass
