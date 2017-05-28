import os
import numpy as np
import soundfile as sf
import torch as t
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

        self.data_path = data_path

        self.first_level_data_folders = [self.data_path + 'dev-clean/' + path
                                         for path in os.listdir(self.data_path + '/dev-clean')
                                         if path != '.DS_Store']
        self.second_level_data_folders = [path + '/' + in_path + '/'
                                          for path in self.first_level_data_folders
                                          for in_path in os.listdir(path)
                                          if in_path != '.DS_Store']
        self.data_files = [path + in_path
                           for path in self.second_level_data_folders
                           for in_path in os.listdir(path)
                           if in_path != '.DS_Store']
        self.audio_files = [file for file in self.data_files if file.endswith('.flac')]
        self.text_files = [file for file in self.data_files if file.endswith('.txt')]

        '''
        go_token (or stop_token) uses to mark start (or end) of the sequence while decoding
        pad_token uses to fill tensor to fixed-size length
        '''
        self.text_go_token = '>'
        self.text_stop_token = '<'
        self.text_pad_token = '_'

        self.audio_go_token = 1.
        self.audio_stop_token = -1.
        self.audio_pad_token = 0

        self.tensors_path = self.data_path + 'preprocessed_data/'
        self.idx_file = self.tensors_path + 'vocab.pkl'
        self.tensor_files = [self.tensors_path + 'train_tensor.npy', self.tensors_path + 'valid_tensor.npy']

        idx_file_exist = os.path.exists(self.idx_file)
        tensor_files_exist = all([os.path.exists(file) for file in self.tensor_files])

        if idx_file_exist and tensor_files_exist and not force_preprocessing:
            print('preprocessed data loading have started')
            self.load_preprocessed_data()
            print('preprocessed data have loaded')
        else:
            print('data preprocessing have started')
            self.preprocess_data()
            print('data have preprocessed')

    def build_character_vocab(self, data):
        """
        :param data: whole data string
        :return: vocab_size - size of unique characters in corpus
                 idx_to_char - array of shape [vocab_size] containing ordered list of inique characters
                 char_to_idx - dictionary of shape [vocab_size]
                        such that idx_to_char[char_to_idx[some_char]] = some_char
                        where some_char is such that idx_to_char contains its idx
        """

        # unique characters with blind symbol
        chars = list(set(data)) + [self.text_go_token, self.text_stop_token, self.text_pad_token]
        vocab_size = len(chars)

        # mappings itself
        idx_to_char = chars
        char_to_idx = {x: i for i, x in enumerate(idx_to_char)}

        return vocab_size, idx_to_char, char_to_idx

    def preprocess_data(self):
        """
        performs data preprocessing
        """

        if not os.path.exists(self.tensors_path):
            os.makedirs(self.tensors_path)

        text_data = [open(path, 'r', encoding='utf-8').read() for path in self.text_files]
        text_data = [line.lower() for file in text_data for line in file.split('\n')[:-1]]
        text_data = [line.split(' ', 1) for line in text_data]

        string_data = ' '.join([line[1] for line in text_data])
        self.vocab_size, self.idx_to_char, self.char_to_idx = self.build_character_vocab(string_data)
        del string_data

        text_data = np.array([{
            'audio_path': self.data_path + 'dev-clean/' + '/'.join(line[0].split('-')[:-1]) + '/' + line[0] + '.flac',
            'text_data': [self.char_to_idx[char] for char in line[1]]
        }
            for line in text_data])
        np.random.shuffle(text_data)

        self.valid_data, self.train_data = text_data[400:], text_data[:400]
        del text_data

        np.save(self.tensor_files[1], self.valid_data)
        np.save(self.tensor_files[0], self.train_data)

        with open(self.idx_file, 'wb') as f:
            cPickle.dump(self.idx_to_char, f)

    def load_preprocessed_data(self):

        self.idx_to_char = cPickle.load(open(self.idx_file, "rb"))
        self.vocab_size = len(self.idx_to_char)
        self.char_to_idx = dict(zip(self.idx_to_char, range(len(self.idx_to_char))))

        [self.train_data, self.valid_data] = [np.load(path) for path in self.tensor_files]

    def next_batch(self, num_batches, target, use_cuda=False):
        """
        :param num_batches: num_batches to lockup from data 
        :param target: if target == 'train' then train data uses as target, in other case test data is used
        :param use_cuda: whether to use cuda
        :return: encoder word and character input, latent representations, its sizes, decoder input and output
        """

        target = 0 if target == 'train' else 1
        data = [self.train_data, self.valid_data][target]
        target_len = len(data)

        indexes = np.array(np.random.randint(target_len, size=num_batches))
        data = [data[idx] for idx in indexes]

        text_input = [element['text_data'] for element in data]
        audio_input = [list(sf.read(element['audio_path'])[0]) for element in data]
        print(data[0]['audio_path'])

        decoder_text_input = [[self.char_to_idx[self.text_go_token]] + element for element in text_input]
        decoder_text_target = [element + [self.char_to_idx[self.text_stop_token]] for element in text_input]

        decoder_audio_input = [[self.audio_go_token] + element for element in audio_input]
        decoder_audio_target = [element + [self.audio_stop_token] for element in audio_input]

        max_text_len = max([len(element) for element in text_input])
        max_audio_len = max([len(element) for element in audio_input])

        for i in range(num_batches):
            text_to_add = max_text_len - len(text_input[i])
            audio_to_add = max_audio_len - len(audio_input[i])

            text_input[i] = text_input[i] + [self.char_to_idx[self.text_pad_token]] * text_to_add
            decoder_text_input[i] = decoder_text_input[i] + [self.char_to_idx[self.text_pad_token]] * text_to_add
            decoder_text_target[i] = decoder_text_target[i] + [self.char_to_idx[self.text_pad_token]] * text_to_add

            audio_input[i] = audio_input[i] + [self.audio_pad_token] * audio_to_add
            decoder_audio_input[i] = decoder_audio_input[i] + [self.audio_pad_token] * audio_to_add
            decoder_audio_target[i] = decoder_audio_target[i] + [self.audio_pad_token] * audio_to_add

        [text_input, decoder_text_input, decoder_text_target,
         audio_input, decoder_audio_input, decoder_audio_target] = \
            np.array(text_input), np.array(decoder_text_input), np.array(decoder_text_target), \
            np.array(audio_input), np.array(decoder_audio_input), np.array(decoder_audio_target)

        [text_input, decoder_text_input, decoder_text_target,
         audio_input, decoder_audio_input, decoder_audio_target] = \
            [Variable(t.from_numpy(var)) for var in [text_input, decoder_text_input, decoder_text_target,
                                                     audio_input, decoder_audio_input, decoder_audio_target]]

        if use_cuda:
            [text_input, decoder_text_input, decoder_text_target,
             audio_input, decoder_audio_input, decoder_audio_target] = \
                [var.cuda() for var in [text_input, decoder_text_input, decoder_text_target,
                                        audio_input, decoder_audio_input, decoder_audio_target]]

        return [text_input, decoder_text_input, decoder_text_target,
                audio_input, decoder_audio_input, decoder_audio_target]

    def sample_character_from_distribution(self, distribution):
        idx = np.random.choice(self.vocab_size, p=distribution.ravel())
        return self.idx_to_char[idx]
