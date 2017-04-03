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

        self.conv_weights = nn.ParameterList([Parameter(t.Tensor(out_c, in_c, kernel_size, kernel_size))
                                              for out_c, in_c, kernel_size in self.params.discr_kernels])
        self.conv_biases = nn.ParameterList([Parameter(t.Tensor(out_c)) for out_c, _, _ in self.params.discr_kernels])

        for weight in self.conv_weights:
            init.xavier_uniform(weight, gain=math.sqrt(2.0))

        self.out_size = (int(512 / (2 ** (len(self.params.discr_kernels) + 1))) ** 2) * self.params.discr_kernels[-1][0]

        self.fc = nn.Linear(self.out_size, 1)

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
        data = self.unroll_convolutions(data)

        result = self.fc(data).squeeze(1)
        del data

        D_fake, D_real = result[:batch_size], result[batch_size:]
        del result

        discriminator_loss = -D_real.mean() + D_fake.mean()
        generator_loss = -D_fake.mean()

        return discriminator_loss, generator_loss

    def unroll_convolutions(self, input):
        [batch_size, _, _, _] = input.size()

        for i in range(len(self.conv_weights)):
            input = F.relu(F.conv2d(input, self.conv_weights[i], self.conv_biases[i], padding=2))
            input = F.avg_pool2d(input, 2) if i < len(self.conv_weights) - 1 else F.avg_pool2d(input, 4)

        return input.view(batch_size, self.out_size)
