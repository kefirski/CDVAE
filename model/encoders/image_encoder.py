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
    def __init__(self, params):
        super(ImageEncoder, self).__init__()

        self.params = params

        self.conv_weights = nn.ParameterList([Parameter(t.Tensor(out_c, in_c, kernel_size, kernel_size))
                                              for out_c, in_c, kernel_size in self.params.encoder_kernels])
        self.conv_biases = nn.ParameterList([Parameter(t.Tensor(out_c)) for out_c, _, _ in self.params.encoder_kernels])

        for weight in self.conv_weights:
            init.xavier_uniform(weight, gain=math.sqrt(2.0))

        self.out_size = self.params.image_encoder_out_size

        self.hw = Highway(self.out_size, 3, F.relu)

    def forward(self, images):
        """
        :param images: Array of image paths to encode from 
        :return: An tensor with shape of [len(images), out_size]
        """

        images = [misc.imread(path) / 255 for path in images]
        images = [(Variable(t.from_numpy(image))).float().transpose(2, 0).contiguous() for image in images]
        images = t.cat([expand_with_zeroes(var, [512, 512]).unsqueeze(0) for var in images], 0)

        images = self.unroll_convolutions(images)

        return self.hw(images)

    def unroll_convolutions(self, input):
        [batch_size, _, _, _] = input.size()

        for i in range(self.params.encoder_conv_num_layers):
            input = F.relu(F.conv2d(input, self.conv_weights[i], self.conv_biases[i], padding=2))
            input = F.avg_pool2d(input, 2) if i < self.params.encoder_conv_num_layers - 1 else F.avg_pool2d(input, 4)

        return input.view(batch_size, self.out_size)
