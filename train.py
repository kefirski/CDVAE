import argparse
import os
import numpy as np
import torch as t
import scipy.misc
from torch.optim import Adam
from utils.batchloader import BatchLoader
from utils.parameters import Parameters
from model.cdvae import CDVAE


if __name__ == "__main__":

    if not os.path.exists('data/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    if not os.path.exists('samplings/'):
        os.makedirs('samplings/')

    path_prefix = ''

    parser = argparse.ArgumentParser(description='CDVAE')
    parser.add_argument('--num-iterations', type=int, default=60000, metavar='NI',
                        help='num iterations (default: 6000)')
    parser.add_argument('--batch-size', type=int, default=2, metavar='BS',
                        help='batch size (default: 2)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    args = parser.parse_args()

    batch_loader = BatchLoader(path_prefix)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    cdvae = CDVAE(parameters, path_prefix)
    if args.use_cuda:
        cdvae = cdvae.cuda()

    s2i_optimizer = Adam(cdvae.seq2image_parameters(), args.learning_rate)
    i2s_optimizer = Adam(cdvae.image2seq_parameters(), args.learning_rate)
    disc_optimizer = Adam(cdvae.discr_parameters(), args.learning_rate)

    train_step = cdvae.trainer(s2i_optimizer, i2s_optimizer, disc_optimizer, batch_loader)

    for iteration in range(args.num_iterations):
        (reconst_loss_s2i, kld_s2i, g_loss_s2i), (reconst_loss_i2s, kld_i2s), kld_id_loss = \
            train_step(args.batch_size, 5, args.use_cuda, args.dropout)

        if iteration % 10 == 0:
            print('\n')
            print('|-----------------------------------------------------------------------|')
            print(iteration)
            print('|---------reconst-loss--------kl-distance-------generation-loss---------|')
            print('|----------------------------------s2i----------------------------------|')
            print(reconst_loss_s2i.data.cpu().numpy()[0],
                  kld_s2i.data.cpu().numpy()[0],
                  g_loss_s2i.data.cpu().numpy()[0])
            print('|----------------------------------i2s----------------------------------|')
            print(reconst_loss_i2s.data.cpu().numpy()[0],
                  kld_i2s.data.cpu().numpy()[0])
            print('|---------------------------------kl-id---------------------------------|')
            print(kld_id_loss.data.cpu().numpy()[0])
            print('|-----------------------------------------------------------------------|')

        if iteration % 25 == 0:
            seed = np.random.normal(size=[1, parameters.latent_variable_size])
            sampled_image, sampled_seq = cdvae.sample(batch_loader, [[450, 450]], 12, seed, args.use_cuda)
            print(sampled_seq)
            scipy.misc.imsave('{}_image.jpg'.format(iteration), sampled_image[0])
            with open('{}_ann.jpg'.format(iteration), 'w') as f:
                f.write(sampled_seq)

