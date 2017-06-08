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

        self.split = 500

        self.data_path = data_path

        self.text_files = [self.data_path + 'ru.txt', self.data_path + 'en.txt']

        '''
        go_token (stop_token) uses to mark start (end) of the sequence while decoding
        pad_token uses to fill tensor to fixed-size length
        '''
        self.go_token = '>'
        self.pad_token = '_'
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

    def build_vocab(self, sentences):
        """
        build_vocab(self, sentences) -> vocab_size, idx_to_word, word_to_idx
            vocab_size - number of unique words in corpus
            idx_to_word - array of shape [vocab_size] containing ordered list of unique words
            word_to_idx - dictionary of shape [vocab_size]
                such that idx_to_char[idx_to_word[some_word]] = some_word
                where some_word is such that idx_to_word contains it
        """

        '''
        Takes unique words in whole sensences 
        and creates from them idx_to_word and word_to_idx objects
        '''

        word_counts = collections.Counter(sentences)

        idx_to_word = [x[0] for x in word_counts.most_common()]
        idx_to_word = list(sorted(idx_to_word)) + [self.pad_token, self.go_token, self.stop_token]

        word_to_idx = {x: i for i, x in enumerate(idx_to_word)}

        words_vocab_size = len(idx_to_word)

        return words_vocab_size, idx_to_word, word_to_idx

    @staticmethod
    def clean_string(string):

        string = re.sub(r"[^가-힣A-Za-zА-Яа-я0-9(),!?:;.\'`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r"\.", " .", string)
        string = re.sub(r",", " ,", string)
        string = re.sub(r":", " : ", string)
        string = re.sub(r"\"", " \" ", string)
        string = re.sub(r"«", "« ", string)
        string = re.sub(r"»", "» ", string)
        string = re.sub(r";", " ;", string)
        string = re.sub(r"!", " !", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " )", string)
        string = re.sub(r"\?", " ?", string)
        string = re.sub(r'([\s])+', ' ', string)

        return string.lower()

    def preprocess_data(self):
        """
        performs data preprocessing
        """

        if not os.path.exists(self.preprocessings_path):
            os.makedirs(self.preprocessings_path)

        data = [open(path, 'r', encoding='utf-8').read() for path in self.text_files]
        data = [[BatchLoader.clean_string(line).split() for line in domain.split('\n')[:-1]] for domain in data]

        self.vocab_size_ru, self.idx_to_word_ru, self.word_to_idx_ru = self.build_vocab(
            [word for line in data[0] for word in line]
        )
        self.vocab_size_en, self.idx_to_word_en, self.word_to_idx_en = self.build_vocab(
            [word for line in data[1] for word in line]
        )

        data[0] = [[self.word_to_idx_ru[word] for word in line] for line in data[0]]
        data[1] = [[self.word_to_idx_en[word] for word in line] for line in data[1]]
        data = np.array(data)

        self.valid_data, self.train_data = [[domain[:self.split] for domain in data],
                                            [domain[self.split:] for domain in data]]
        self.valid_data = np.array(self.valid_data)
        self.train_data = np.array(self.train_data)

        self.data_len = [len(self.train_data[0]), len(self.valid_data[0])]

        for i, path in enumerate(self.tensor_files):
            np.save(path, [self.train_data, self.valid_data][i])

        for i, file in enumerate(self.idx_files):
            with open(file, 'wb') as f:
                cPickle.dump([self.idx_to_word_ru, self.idx_to_word_en][i], f)

    def load_preprocessed_data(self):

        self.idx_to_word_ru, self.idx_to_word_en = (cPickle.load(open(path, "rb")) for path in self.idx_files)

        self.vocab_size_ru, self.vocab_size_en = (len(idx) for idx in [self.idx_to_word_ru, self.idx_to_word_en])

        self.word_to_idx_ru, self.word_to_idx_en = (dict(zip(idx, range(len(idx))))
                                                    for idx in [self.idx_to_word_ru, self.idx_to_word_en])

        self.train_data, self.valid_data = (np.load(path) for path in self.tensor_files)
        self.data_len = [len(self.train_data[0]), len(self.valid_data[0])]

    @staticmethod
    def bag_window(seq, window=5):

        assert window > 1 and isinstance(window, int)

        context = []
        seq_len = len(seq)
        num_slices = seq_len - window + 1

        for i in range(num_slices):
            context += [seq[i:i + window]]

        return [[value] + c[:i] + c[i + 1:] for c in context for i, value in enumerate(c)]

    def next_batch(self, batch_size, target: str, use_cuda=False):
        """
        :param batch_size: num_batches to lockup from data
        :param target: if target == 'train' then train data uses as target, in other case test data is used
        :param use_cuda: whether to use cuda
        :return: encoder word and character input, latent representations, its sizes, decoder input and output
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

    def next_embedding_batch(self, batch_size, lang):
        """
        :param batch_size: batch size
        :param lang: which vocabulary to use
        :return: Arrays of input and target for embedding learning
        """

        """
        Sample indexes in order to choose random pairs of words and contexts 
        """

        lang = 0 if lang == 'ru' else 1
        data_len = len(self.train_data[lang])

        result = []

        while len(result) != batch_size:
            index = np.array(np.random.randint(data_len, size=1))
            line = self.train_data[lang, index[0]]

            pairs = BatchLoader.bag_window(line)

            neccesary_to_add = batch_size - len(result)

            if len(pairs) >= neccesary_to_add:
                index = np.array(np.random.randint(len(pairs), size=neccesary_to_add))
                pairs = [pairs[i] for i in index]

            result += pairs

        result = np.array(result)

        return result[:, 0], result[:, 1:]

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

        lang = 0 if lang == 'ru' else 1
        word_to_idx = [self.word_to_idx_ru, self.word_to_idx_en][lang]

        '''Add go token before decoder input and stop token after decoder target'''
        decoder_input = [[word_to_idx[self.go_token]] + line for line in encoder_input]
        decoder_target = [line + [word_to_idx[self.stop_token]] for line in encoder_input]

        '''Evaluate how much it is necessary to fill with pad tokens to make the same lengths'''
        input_seq_len = [len(line) for line in encoder_input]
        max_input_seq_len = max(input_seq_len)
        to_add = [max_input_seq_len - len(encoder_input[i]) for i in range(batch_size)]

        for i in range(batch_size):
            encoder_input[i] += [word_to_idx[self.pad_token]] * to_add[i]
            decoder_input[i] += [word_to_idx[self.pad_token]] * to_add[i]
            decoder_target[i] += [word_to_idx[self.pad_token]] * to_add[i]

        result = [np.array(var) for var in [encoder_input, decoder_input, decoder_target]]
        result = [Variable(t.from_numpy(var)).long() for var in result]
        if use_cuda:
            result = [var.cuda() for var in result]

        return tuple(result)

    def sample_word(self, distribution, lang: str):
        """
        :param distribution: An array of probabilities
        :param lang: which vocabulary to use
        :return: An index of sampled from distribution word
        """

        lang = 0 if lang == 'ru' else 1
        vocab_size = [self.vocab_size_ru, self.vocab_size_en][lang]

        return np.random.choice(vocab_size, p=distribution.ravel())
