import torch as t
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F


class Disriminator(nn.module):
    def __init__(self, params):
        """
        discriminator network for we 
        """
        super(Disriminator, self).__init__()

        self.params = params

        self.conv_weights = nn.ParameterList([Parameter(t.Tensor(out_c, in_c, kernel_size))
                                              for out_c, in_c, kernel_size in self.params.discr_kernels])
        self.conv_biases = nn.ParameterList([Parameter(t.Tensor(out_c)) for out_c, _, _ in self.params.discr_kernels])

        for weight in self.conv_weights:
            init.xavier_uniform(weight, gain=math.sqrt(2.0))
