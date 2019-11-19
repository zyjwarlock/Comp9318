import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class Data_process:

    def __init__(self, class_0_path, class_1_path, threshold=0):
        with open(class_0_path, 'r') as class0:
            class_0 = [line.strip().split(' ') for line in class0]
        with open(class_1_path, 'r') as class1:
            class_1 = [line.strip().split(' ') for line in class1]

        train_set = [" ".join(e) for e in class_0] + [" ".join(e) for e in class_1]
        self._output = ['0' for e in range(len(class_0))] + ['1' for e in range(len(class_1))]
        vectorizer = CountVectorizer(token_pattern='\S+', min_df=threshold)
        self._inputs = vectorizer.fit_transform(train_set)
        self._vocab_size = len(vectorizer.vocabulary_)
        transformer = TfidfTransformer()
        self._inputs = transformer.fit_transform(self._inputs).todense()
        self._num_timestep = self._inputs.shape[1]
        self._output = np.asarray(self._output, dtype=np.int32)
        self._random_shuffle()
        self._indicator = 0

    def vocab_size(self):
        return self._vocab_size

    def num_classes(self):
        return 2

    def num_timestep(self):
        return self._num_timestep

    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        self._inputs = self._inputs[p]
        self._outputs = self._output[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > len(self._inputs):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > len(self._inputs):
            raise Exception("batch_size: %d is too large" % batch_size)

        batch_inputs = self._inputs[self._indicator: end_indicator]
        batch_outputs = self._outputs[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_outputs



class Vocab:

    def __init__(self, lines, num_word_threshold):
        self._word_to_id = {}
        self._unk = -1
        self._num_word_threshold = num_word_threshold
        self._read_dict(lines)

    def _read_dict(self, lines):
        for line in lines:
            #word, frequency = line.strip('\r\n').split('\t')
            word, frequency =line, lines[line]
            word = word.decode('utf-8')
            frequency = int(frequency)
            if frequency < self._num_word_threshold:
                continue
            idx = len(self._word_to_id)
            if word == '<UNK>':
                self._unk = idx
            self._word_to_id[word] = idx

    def word_to_id(self, word):
        return self._word_to_id.get(word, self._unk)

    @property
    def unk(self):
        return self._unk

    def size(self):
        return len(self._word_to_id)

    def sentence_to_id(self, sentence):
        word_ids = [self._word_to_id[cur_word] for cur_word in sentence.split()]
        return word_ids

class CategoryDict:
    def __init__(self, lines):
        self._category_to_id = {}
        # with open(filename, 'r') as f:
        #     lines = f.readlines()
        # for line in lines:
        #     category = line.strip('\r\n').decode('utf-8')
        #     idx = len(self._category_to_id)
        #     self._category_to_id[category] = idx
        for line in lines:
            category = line
            idx = len(self._category_to_id)
            self._category_to_id[category] = idx

    def size(self):
        return len(self._category_to_id)
    def category_to_id(self, category):
        if not category in self._category_to_id:
            raise Exception(
                "%s is not in our category list" % category)
        return self._category_to_id[category]

class TextDataSet:
    def __init__(self, filename, vocab, category_vocab, num_timesteps):
        self._vocab = vocab
        self._category_vocab = category_vocab
        self._num_timesteps = num_timesteps
        self._inputs = []
        self._outputs = []
        self._indicator = 0
        self._parse_file(filename)

    def _parse_file(self, filename):
        tf.logging.info('Loading data from %s', filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines():
            label, content = line.strip('\r\n').decode('utf-8').split('\t')
            id_label = self._category_vocab.category_to_id(label)
            id_words = self._vocab.sentence_to_id(content)
            id_words = id_words[0: self._num_timesteps]
            padding_num = self._num_timesteps - len(id_words)
            id_words = id_words + [self._vocab.unk for i in range(padding_num)]
            self._inputs.append(id_words)
            self._outputs.append(id_label)

        self._inputs = np.asarray(self._inputs, dtype = np.int32)
        self._outputs = np.asarray(self._outputs, dtype = np.int32)
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        self._inputs = self._inputs[p]
        self._outputs = self._outputs[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > len(self._inputs):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > len(self._inputs):
            raise Exception("batch_size: %d is too large" % batch_size)

        batch_inputs = self._inputs[self._indicator: end_indicator]
        batch_outputs = self._outputs[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_outputs