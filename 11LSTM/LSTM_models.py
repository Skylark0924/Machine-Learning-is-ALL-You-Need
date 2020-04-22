import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # 隐藏warning

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import collections

class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y

class Keras_LSTM():
    def __init__(self, vocabulary, hidden_size, num_steps, use_dropout=True):
        super().__init__()
        self.vocabulary = vocabulary
        self.num_steps = num_steps
        self.model = Sequential()
        self.model.add(Embedding(vocabulary, hidden_size, input_length=self.num_steps))
        self.model.add(LSTM(hidden_size, return_sequences=True))
        self.model.add(LSTM(hidden_size, return_sequences=True))
        if use_dropout:
            self.model.add(Dropout(0.5))
        self.model.add(TimeDistributed(Dense(self.vocabulary)))
        self.model.add(Activation('softmax'))
        self.model.summary()
    
    def fit(self, train_data, valid_data, batch_size, num_epochs):
        checkpointer = ModelCheckpoint('./log/Keras_LSTM/model-{epoch:02d}.hdf5', verbose=1)
        train_data_generator = KerasBatchGenerator(train_data, self.num_steps, batch_size, self.vocabulary,
                                           skip_step=self.num_steps)
        valid_data_generator = KerasBatchGenerator(valid_data, self.num_steps, batch_size, self.vocabulary,
                                           skip_step=self.num_steps)
        # initiate Adam optimizer
        opt = keras.optimizers.Adam()
        # Let's train the model using Adam
        self.model.compile(loss='categorical_crossentropy', optimizer=opt , metrics=['categorical_accuracy'])
        self.model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*self.num_steps), num_epochs,
                    validation_data=valid_data_generator.generate(),
                    validation_steps=len(valid_data)//(batch_size*self.num_steps), callbacks=[checkpointer])
    
    def evaluate(self, test_data, reversed_dictionary):
        example_test_generator = KerasBatchGenerator(test_data, self.num_steps, 1, self.vocabulary,
                                                     skip_step=1)
        dummy_iters = 40
        num_predict = 10
        true_print_out = "Actual words: "
        pred_print_out = "Predicted words: "
        for i in range(num_predict):
            data = next(example_test_generator.generate())
            prediction = self.model.predict(data[0])
            predict_word = np.argmax(prediction[:, self.num_steps - 1, :])
            true_print_out += reversed_dictionary[test_data[self.num_steps + dummy_iters + i]] + " "
            pred_print_out += reversed_dictionary[predict_word] + " "
        print(true_print_out)
        print(pred_print_out)

class Torch_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        torch.manual_seed(1)
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    


def keras_data():
    data_path = './dataset/PTB_data'
    # get the data paths
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(train_data[:5])
    print(word_to_id)
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]
