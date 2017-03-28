import math
import torch as t
import torch.nn as nn
import torch.nn.init as init
from ..other.corner_unpool import Corner2dMaxUnpool


class SeqDeconv(nn.Module):
    def __init__(self, params):
        super(SeqDeconv, self).__init__()

        self.params = params

        self.deconvolutions = nn.ModuleList([nn.ConvTranspose2d(in_chan, out_chan, kernel_size, bias=True, padding=2)
                                             for in_chan, out_chan, kernel_size, _ in self.params.deconv_kernels])
        [init.xavier_uniform(deconv.weight, gain=math.sqrt(2.0)) for deconv in self.deconvolutions]

        self.max_unpool = Corner2dMaxUnpool(kernel_size=2)

    def forward(self, x, out_size):
        input_size = x.size()
        assert len(input_size) == 4, 'Invalid input rang. Must be equal to 4, but {} found'.format(len(input_size))
        [batch_size, _, _, _] = input_size

        for i in range(self.params.deconv_num_layers):
            _, out_chan, _, (out_h, out_w) = self.params.deconv_kernels[i]

            x = self.max_unpool(x)
            x = self.deconvolutions[i](x, output_size=[batch_size, out_chan, out_h, out_w])

        return x
