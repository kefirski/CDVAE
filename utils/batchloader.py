import os
import re
import numpy as np
import torch as t
import collections
from torch.autograd import Variable
from six.moves import cPickle


class BatchLoader:
    def __init__(self, data_path='./data/', force_preprocessing=False):
        """
        :param data_path: string prefix to path of data folder
        :param force_preprocessing: whether to force data preprocessing
        """

        assert isinstance(data_path, str), \
            'Invalid data_path_prefix type. Required {}, but {} found'.format(str, type(data_path))

        self.split = 3000

        self.data_path = data_path

        self.text_files = [self.data_path + 'ru.txt', self.data_path + 'en.txt']

        '''
        go_token (stop_token) uses to mark start (end) of the sequence while decoding
        pad_token uses to fill tensor to fixed-size length
        '''
        self.go_token = '>'
        self.pad_token = ''
        self.stop_token = '<'

        self.preprocessings_path = self.data_path + 'preprocessed_data/'

        self.idx_files = [self.preprocessings_path + 'vocab_ru.pkl',
                          self.preprocessings_path + 'vocab_en.pkl']

        self.tensor_files = [self.preprocessings_path + 'train_tensor.npy',
                             self.preprocessings_path + 'valid_tensor.npy']

        idx_files_exist = all([os.path.exists(file) for file in self.idx_files])
        tensor_files_exist = all([os.path.exists(file) for file in self.tensor_files])

        if idx_files_exist and tensor_files_exist and not force_preprocessing:

            print('preprocessed data loading have started')

            self.load_preprocessed_data()

            print('preprocessed data have loaded')
        else:

            print('data preprocessing have started')

            self.preprocess_data()

            print('data have preprocessed')

    def build_vocab(self, data):

        # unique characters with blind symbol
        chars = list(set(data)) + [self.pad_token, self.go_token, self.stop_token]
        chars_vocab_size = len(chars)

        # mappings itself
        idx_to_char = chars
        char_to_idx = {x: i for i, x in enumerate(idx_to_char)}

        return chars_vocab_size, idx_to_char, char_to_idx

    def preprocess_data(self):
        """
        performs data preprocessing
        """

        if not os.path.exists(self.preprocessings_path):
            os.makedirs(self.preprocessings_path)

        data = [open(path, 'r', encoding='utf-8').read().lower() for path in self.text_files]

        v_s_ru, i_to_c_ru, c_to_i_ru = self.build_vocab(data[0])
        v_s_en, i_to_c_en, c_to_i_en = self.build_vocab(data[1])

        self.vocab_size = {'ru': v_s_ru, 'en': v_s_en}
        self.idx_to_char = {'ru': i_to_c_ru, 'en': i_to_c_en}
        self.char_to_idx = {'ru': c_to_i_ru, 'en': c_to_i_en}

        '''Take every line and change characters with indexes'''
        for i, lang in enumerate(['ru', 'en']):
            data[i] = [[self.char_to_idx[lang][char] for char in line] for line in data[i].split('\n')[:-1]]

        self.valid_data, self.train_data = [domain[:self.split] for domain in data],\
                                           [domain[self.split:] for domain in data]

        self.valid_data = np.array(self.valid_data)
        self.train_data = np.array(self.train_data)

        self.data_len = [len(self.train_data[0]), len(self.valid_data[0])]

        for i, path in enumerate(self.tensor_files):
            np.save(path, [self.train_data, self.valid_data][i])

        for i, file in enumerate(self.idx_files):
            with open(file, 'wb') as f:
                cPickle.dump([self.idx_to_char['ru'], self.idx_to_char['en']][i], f)

    def load_preprocessed_data(self):

        i_to_c_ru, i_to_c_en = (cPickle.load(open(path, "rb")) for path in self.idx_files)
        v_s_ru, v_s_en = (len(idx) for idx in [i_to_c_ru, i_to_c_en])
        c_to_i_ru, c_to_i_en = (dict(zip(idx, range(len(idx)))) for idx in [i_to_c_ru, i_to_c_en])

        self.vocab_size = {'ru': v_s_ru, 'en': v_s_en}
        self.idx_to_char = {'ru': i_to_c_ru, 'en': i_to_c_en}
        self.char_to_idx = {'ru': c_to_i_ru, 'en': c_to_i_en}

        self.train_data, self.valid_data = (np.load(path) for path in self.tensor_files)
        self.data_len = [len(self.train_data[0]), len(self.valid_data[0])]

    def next_batch(self, batch_size, target: str, use_cuda=False):
        """
        :param batch_size: num_batches to lockup from data
        :param target: if target == 'train' then train data uses as target, in other case test data is used
        :param use_cuda: whether to use cuda
        :return: encoder and decoder input
        """

        """
        Randomly takes batch_size of lines from target data 
        and wrap them into ready to feed in the model Tensors
        """

        target = 0 if target == 'train' else 1

        indexes = np.array(np.random.randint(self.data_len[target], size=batch_size))
        data = [self.train_data, self.valid_data][target]

        encoder_input_ru, encoder_input_en = ([np.copy(data[i, idx]).tolist() for idx in indexes] for i in range(2))

        return self._wrap_tensor(encoder_input_ru, 'ru', use_cuda), self._wrap_tensor(encoder_input_en, 'en', use_cuda)

    def _wrap_tensor(self, encoder_input, lang: str, use_cuda: bool):
        """
        :param encoder_input: An list of batch size len filled with lists of input indexes
        :param lang: which vocabulary to use
        :param use_cuda: whether to use cuda
        :return: encoder_input, decoder_input and decoder_target tensors of Long type
        """

        """
        Creates decoder input and target from encoder input 
        and fills it with pad tokens in order to initialize Tensors
        """

        batch_size = len(encoder_input)

        '''Add go token before decoder input and stop token after decoder target'''
        decoder_input = [[self.char_to_idx[lang][self.go_token]] + line for line in encoder_input]
        decoder_target = [line + [self.char_to_idx[lang][self.stop_token]] for line in encoder_input]

        '''Evaluate how much it is necessary to fill with pad tokens to make the same lengths'''
        input_seq_len = [len(line) for line in encoder_input]
        max_input_seq_len = max(input_seq_len)
        to_add = [max_input_seq_len - len(encoder_input[i]) for i in range(batch_size)]

        for i in range(batch_size):
            encoder_input[i] += [self.char_to_idx[lang][self.pad_token]] * to_add[i]
            decoder_input[i] += [self.char_to_idx[lang][self.pad_token]] * to_add[i]
            decoder_target[i] += [self.char_to_idx[lang][self.pad_token]] * to_add[i]

        result = [np.array(var) for var in [encoder_input, decoder_input, decoder_target]]
        result = [Variable(t.from_numpy(var)).long() for var in result]
        if use_cuda:
            result = [var.cuda() for var in result]

        return tuple(result)

    def go_input(self, batch_size, lang, use_cuda):

        go_input = np.array([[self.char_to_idx[lang][self.go_token]]] * batch_size)
        go_input = Variable(t.from_numpy(go_input)).long()

        if use_cuda:
            go_input = go_input.cuda()

        return go_input

    @staticmethod
    def sample_character(distribution):
        """
        :param distribution: An array of probabilities
        :return: An index of sampled from distribution character
        """

        return np.random.choice(len(distribution), p=distribution.ravel())
