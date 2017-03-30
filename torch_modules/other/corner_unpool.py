import torch as t
import torch.nn as nn
from torch.autograd import Variable


class Corner2dMaxUnpool(nn.Module):
    def __init__(self, kernel_size):
        super(Corner2dMaxUnpool, self).__init__()

        self.kernel_size = kernel_size
        self.unpool = nn.MaxUnpool2d(kernel_size)

    def forward(self, input):
        """
        :param input: tensor with shape of [batch_size, input_channels, in_height, in_width] 
        :return: unpooled tensor with elements in right bottom corners of output 'cells'
        """

        input_size = input.size()
        assert len(input_size) == 4, 'Invalid input rang. Must be equal to 4, but {} found'.format(len(input_size))
        [batch_size, channels, height, width] = input_size

        assert all([size % self.kernel_size == 0 for size in [height, width]]), 'Invalid kernel size.'

        new_width = width * self.kernel_size
        new_height = height * self.kernel_size

        indexes = Variable(t.LongTensor([i * new_width + j
                                         for i in range(self.kernel_size - 1, new_height, self.kernel_size)
                                         for j in range(self.kernel_size - 1, new_width, self.kernel_size)])) \
            .view(height, width)
        indexes = indexes.repeat(batch_size, channels, 1, 1)

        if input.is_cuda:
            indexes = indexes.cuda()

        result = self.unpool(input, indexes)

        return result
