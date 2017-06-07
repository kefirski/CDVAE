import argparse
import numpy as np
import torch as t
from torch.autograd import Variable
from torch.optim import Adagrad
from utils.batchloader import BatchLoader
from utils.parameters import Parameters
from torch_modules.losses.glove import GloVe

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='word2vec')
    parser.add_argument('--num-iterations', type=int, default=100, metavar='NI',
                        help='num iterations (default: 50000000)')
    parser.add_argument('--batch-size', type=int, default=40, metavar='BS',
                        help='batch size (default: 40)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--lang', type=str, default='ru', metavar='RU',
                        help='language to train (default: ru)')
    args = parser.parse_args()

    batch_loader = BatchLoader()
    params = Parameters(batch_loader.vocab_size_ru, batch_loader.vocab_size_en)

    lang_idx = 0 if args.lang == 'ru' else 1

    glove = GloVe(co_oc=[batch_loader.co_occurence_ru, batch_loader.co_occurence_en][lang_idx],
                  embed_size=params.embed_size)

    if args.use_cuda:
        glove = glove.cuda()

    optimizer = Adagrad(glove.parameters(), 0.05)

    for iteration in range(args.num_iterations):

        input, target = batch_loader.next_embedding_seq(args.batch_size, args.lang)

        loss = glove(input, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 500 == 0:
            loss = loss.cpu().data.numpy()[0]
            print('iteration = {}, loss = {}'.format(iteration, loss))

    word_embeddings = glove.embeddings()
    np.save('data/preprocessed_data/word_embeddings.npy', word_embeddings)