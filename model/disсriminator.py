import math
import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from scipy import misc
from torch.nn import Parameter
from torch.autograd import Variable
from torch_modules.other.expand_with_zeros import expand_with_zeroes


class Disсriminator(nn.Module):
    def __init__(self, params, path_prefix):
        super(Disсriminator, self).__init__()

        self.params = params

        self.path_prefix = path_prefix

        self.main_conv = nn.Sequential(
            # [3, 512, 512] -> [8, 256, 256]
            nn.Conv2d(3, 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # [8, 256, 256] -> [16, 128, 128]
            nn.Conv2d(8, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # [16, 128, 128] -> [32, 64, 64]
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # [32, 64, 64] -> [64, 32, 32]
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # [64, 32, 32] -> [128, 16, 16]
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # [128, 16, 16] -> [256, 4, 4]
            nn.Conv2d(128, 256, 4, 4, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # [256, 4, 4] -> [1, 1, 1]
            nn.Conv2d(256, 1, 4, 4, 1, bias=False)
        )

    def forward(self, generated_data, true_data):
        """
        :param generated_data: An array of tensors with shape [3, height_i, width_i] of batch_size length
        :param true_data: An array of paths of true data
        :return: discrimination result of generated and true data
        """

        assert len(generated_data) == len(true_data), 'generated and true data should have the same length'
        batch_size = len(generated_data)

        true_data = [misc.imread(self.path_prefix + path) / 255 for path in true_data]
        true_data = [(Variable(t.from_numpy(image))).float().transpose(2, 0).contiguous() for image in true_data]

        data = generated_data + true_data
        del true_data
        data = t.cat([expand_with_zeroes(var, [512, 512]).unsqueeze(0) for var in data], 0)
        data = self.main_conv(data).view(-1, 1)

        D_fake, D_real = data[:batch_size], data[batch_size:]
        del data

        discriminator_loss = -D_real.mean() + D_fake.mean()
        generator_loss = -D_fake.mean()

        return discriminator_loss, generator_loss


