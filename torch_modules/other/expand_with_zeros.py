import torch as t
import torch.nn as nn
from torch.autograd import Variable


def expand_with_zeroes(result, size):
    assert len(result.size()) == 3 and len(size) == 2

    [batch_size, in_height, in_width] = result.size()
    [out_height, out_width] = size

    assert out_height >= in_height and out_width >= in_width

    expand_height, expand_width = out_height - in_height, out_width - in_width
    [(expand_height, from_h), (expand_width, from_w)] = [(int(size / 2), 0) if size % 2 == 0
                                                         else (int((size + 1) / 2), 1)
                                                         for size in [expand_height, expand_width]]

    zeros = Variable(t.zeros([batch_size, in_height, expand_width]))
    result = t.cat(tuple([zeros, result, zeros]), 2)
    zeros = Variable(t.zeros([batch_size, expand_height, expand_width * 2 + in_width]))
    result = t.cat(tuple([zeros, result, zeros]), 1)[:, from_h:, from_w:]

    return result
