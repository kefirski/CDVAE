import math
import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from scipy import misc
from torch.nn import Parameter
from torch.autograd import Variable
from torch_modules.other.highway import Highway
from torch_modules.other.expand_with_zeros import expand_with_zeroes


class ImageEncoder(nn.Module):
    def __init__(self, params, path_prefix):
        super(ImageEncoder, self).__init__()

        self.params = params
        self.path_prefix = path_prefix

        self.main_conv = nn.Sequential(
            # [3, 512, 512] -> [8, 256, 256]
            nn.Conv2d(3, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            # [8, 256, 256] -> [16, 128, 128]
            nn.Conv2d(8, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # [16, 128, 128] -> [32, 64, 64]
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # [32, 64, 64] -> [64, 32, 32]
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # [64, 32, 32] -> [128, 16, 16]
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # [128, 16, 16] -> [256, 4, 4]
            nn.Conv2d(128, 256, 4, 4, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # [256, 4, 4] -> [512, 1, 1]
            nn.Conv2d(256, 512, 4, 4, 1, bias=False)
        )

        self.hw = Highway(512, 3, F.relu)

    def forward(self, images):
        """
        :param images: Array of image paths to encode from 
        :return: An tensor with shape of [len(images), out_size]
        """

        batch_size = len(images)

        images = [misc.imread(self.path_prefix + path) / 255 for path in images]
        images = [(Variable(t.from_numpy(image))).float().transpose(2, 0).contiguous() for image in images]
        images = t.cat([expand_with_zeroes(var, [512, 512]).unsqueeze(0) for var in images], 0)

        images = self.main_conv(images).view(batch_size, 512)

        return self.hw(images)

