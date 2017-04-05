import math
import torch as t
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from torch_modules.other.corner_unpool import Corner2dMaxUnpool


class ImageDecoder(nn.Module):
    def __init__(self, params):
        super(ImageDecoder, self).__init__()

        self.params = params

        self.main_deconv = nn.Sequential(
            # [latent_variable_size, 1, 1] -> [1024, 4, 4]
            nn.ConvTranspose2d(self.params.latent_variable_size, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # [1024, 4, 4] -> [512, 8, 8]
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # [512, 8, 8] -> [256, 16, 16]
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # [256, 16, 16] -> [128, 32, 32]
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # [128, 32, 32] -> [64, 64, 64]
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # [64, 64, 64] -> [32, 128, 128]
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # [32, 128, 128] -> [16, 256, 256]
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # [16, 256, 256] -> [8, 512, 512]
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.last_kernel = Parameter(t.Tensor(8, 3, 5, 5))
        init.xavier_uniform(self.last_kernel, gain=math.sqrt(2.0))

        '''
            out of last deconv is [512, 512] image. 
            thus it is neccesary to padd it with base padding to emit base [500, 500] image
        '''
        self.base_padding = 6 + 2

    def forward(self, x, out_size):
        """
        :param x: input tensor with shape of [batch_size, latent_variable_size] 
        :param out_size: array = [out_height, out_width]
        :return An tensor with shape [3, out_height, out_width]
        """

        input_size = x.size()
        assert len(input_size) == 1, 'Invalid input rang. Must be equal to 1, but {} found'.format(len(input_size))
        x = x.view(1, self.params.latent_variable_size, 1, 1)

        assert len(out_size) == 2, \
            'Invalid out_size format. len(out_size) must be equal to 2, but {} found'.format(len(out_size))
        [out_height, out_width] = out_size
        (h_padding, h_even), (w_padding, w_even) = self.padding(out_height), self.padding(out_width)

        x = self.main_deconv(x)

        # final deconv to emit output with given size
        x = F.conv_transpose2d(x, self.last_kernel, padding=(h_padding, w_padding))

        # cropp image if size is odd
        x = x if h_even else x[:, :, :-1, :]
        x = x if w_even else x[:, :, :, :-1]

        return F.tanh(x.squeeze(0))

    def padding(self, size):
        diff = 500 - size
        even = diff % 2 == 0

        return int(self.base_padding + (diff / 2 if even else (diff - 1) / 2)), even
