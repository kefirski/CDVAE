import argparse
import os
import numpy as np
import torch as t
import torch.nn.functional as F
import scipy.misc
from torch.optim import Adam, SGD
from utils.batchloader import BatchLoader
from utils.parameters import Parameters
from model.cdvae import CDVAE

if __name__ == "__main__":

    if not os.path.exists('samplings/'):
        os.makedirs('samplings/')

    path_prefix = './data/'

    parser = argparse.ArgumentParser(description='CDVAE')
    parser.add_argument('--num-iterations', type=int, default=450000, metavar='NI',
                        help='num iterations (default: 450000)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                        help='batch size (default: 10)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--text-dropout', type=float, default=0.1, metavar='TDR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--audio-dropout', type=float, default=0.45, metavar='ADR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--text-save', type=str, default=None, metavar='TS',
                        help='path where save trained text model to (default: None)')
    parser.add_argument('--audio-save', type=str, default=None, metavar='AS',
                        help='path where save trained audio model to (default: None)')
    args = parser.parse_args()

    batch_loader = BatchLoader(path_prefix)
    parameters = Parameters(batch_loader.vocab_size)

    cdvae = CDVAE(parameters)
    if args.use_cuda:
        cdvae = cdvae.cuda()

    optimizer_text = Adam(cdvae.text_vae.parameters(), args.learning_rate)
    optimizer_audio = Adam(cdvae.audio_vae.parameters(), args.learning_rate)

    for iteration in range(args.num_iterations):

        [input_text, dec_input_text, dec_target_text, input_audio, dec_input_audio, dec_target_audio] = \
            batch_loader.next_batch(args.batch_size, 'train', args.use_cuda)

        '''losses from cdvae is tuples of text and audio losses respectively'''
        loss_text, loss_audio = cdvae(args.text_dropout, args.audio_dropout,
                                      input_text, input_audio,
                                      dec_input_text, dec_input_audio,
                                      dec_target_text, dec_target_audio,
                                      iteration)

        optimizer_text.zero_grad()
        loss_text[0].backward(retain_variables=True)
        optimizer_text.step()

        optimizer_audio.zero_grad()
        loss_audio[0].backward()
        optimizer_audio.step()

        if iteration % 10 == 0:
            print('\n')
            print('|-----------------------------------------------------------------------|')
            print(iteration)
            print('|---------reconst-loss-----------kl-loss-----------cd-loss--------------|')
            print('|---------------------------------text----------------------------------|')
            print(loss_text[1].data.cpu().numpy()[0],
                  loss_text[2].data.cpu().numpy()[0],
                  loss_text[3].data.cpu().numpy()[0])
            print('|---------------------------------audio---------------------------------|')
            print(loss_audio[1].data.cpu().numpy()[0],
                  loss_audio[2].data.cpu().numpy()[0],
                  loss_audio[3].data.cpu().numpy()[0])
            print('|-----------------------------------------------------------------------|')
