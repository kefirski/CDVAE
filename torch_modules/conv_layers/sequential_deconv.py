import math
import torch as t
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from ..other.corner_unpool import Corner2dMaxUnpool


class SeqDeconv(nn.Module):
    def __init__(self, params):
        super(SeqDeconv, self).__init__()

        self.params = params

        self.deconvolutions = nn.ModuleList([nn.ConvTranspose2d(in_chan, out_chan, kernel_size, bias=True, padding=2)
                                             for in_chan, out_chan, kernel_size, _ in self.params.deconv_kernels])
        for deconv in self.deconvolutions:
            init.xavier_uniform(deconv.weight, gain=math.sqrt(2.0))

        last_in, last_out, last_kernel_size, _ = self.params.last_kernel
        self.last_kernel = Parameter(t.Tensor(last_in, last_out, last_kernel_size, last_kernel_size))
        self.last_bias = Parameter(t.Tensor(last_out))
        init.xavier_uniform(self.last_kernel, gain=math.sqrt(2.0))

        self.max_unpool = Corner2dMaxUnpool(kernel_size=2)

        '''
            out of last deconv is [516, 516] image. 
            thus it is neccesary to padd it with base padding to emit standart [500, 500] image
        '''
        self.base_padding = 8

    def forward(self, x, out_size):
        """
        :param x: input tensor with shape of [batch_size, input_channels, in_height, in_width] 
        :param out_size: array of [out_height, out_width]
        """

        input_size = x.size()
        assert len(input_size) == 4, 'Invalid input rang. Must be equal to 4, but {} found'.format(len(input_size))
        [batch_size, _, _, _] = input_size
        assert len(out_size) == 2, \
            'Invalid out_size format. len(out_size) must be equal to 2, but {} found'.format(len(out_size))
        [out_height, out_width] = out_size
        (h_padding, h_even), (w_padding, w_even) = self.padding(out_height), self.padding(out_width)

        for i in range(self.params.deconv_num_layers):
            _, out_chan, _, (out_h, out_w) = self.params.deconv_kernels[i]

            x = self.max_unpool(x)
            x = F.relu(self.deconvolutions[i](x, output_size=[batch_size, out_chan, out_h, out_w]))

        # final deconv to emit output with given size
        x = self.max_unpool(x)
        x = F.conv_transpose2d(x, self.last_kernel, self.last_bias, padding=(h_padding, w_padding))

        # cropp image if size is odd
        x = x if h_even else x[:, :, :-1, :]
        x = x if w_even else x[:, :, :, :-1]

        return x

    def padding(self, size):
        diff = 500 - size
        even = diff % 2 == 0

        return int(self.base_padding + (diff / 2 if even else (diff - 1) / 2)), even
