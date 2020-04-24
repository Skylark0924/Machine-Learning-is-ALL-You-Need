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

import sys
sys.path.append("/home/skylark/Github/Machine-Learning-Basic-Codes")
from utils.tool_func import *

class Skylark_LSTM():
    def __init__(self):
        self.H = 128 # Number of LSTM layer's neurons
        self.D = 10000 # Number of input dimension == number of items in vocabulary
        Z = self.H + self.D # Because we will concatenate LSTM state with the input

        self.model = dict(
            Wf=np.random.randn(Z, self.H) / np.sqrt(Z / 2.),
            Wi=np.random.randn(Z, self.H) / np.sqrt(Z / 2.),
            Wc=np.random.randn(Z, self.H) / np.sqrt(Z / 2.),
            Wo=np.random.randn(Z, self.H) / np.sqrt(Z / 2.),
            Wy=np.random.randn(self.H, self.D) / np.sqrt(self.D / 2.),
            bf=np.zeros((1, self.H)),
            bi=np.zeros((1, self.H)),
            bc=np.zeros((1, self.H)),
            bo=np.zeros((1, self.H)),
            by=np.zeros((1, self.D))
        )

    def lstm_forward(self, X, state):
        m = self.model
        Wf, Wi, Wc, Wo, Wy = m['Wf'], m['Wi'], m['Wc'], m['Wo'], m['Wy']
        bf, bi, bc, bo, by = m['bf'], m['bi'], m['bc'], m['bo'], m['by']

        h_old, c_old = state

        # One-hot encode
        X_one_hot = np.zeros(self.D)
        X_one_hot[X] = 1.
        X_one_hot = X_one_hot.reshape(1, -1)
        # X = np.array(X).reshape(1, -1)

        # Concatenate old state with current input
        X = np.column_stack((h_old, X_one_hot))

        hf = sigmoid(X @ Wf + bf)
        hi = sigmoid(X @ Wi + bi)
        ho = sigmoid(X @ Wo + bo)
        hc = tanh(X @ Wc + bc)

        c = hf * c_old + hi * hc
        h = ho * tanh(c)

        y = h @ Wy + by
        prob = softmax(y)

        state = (h, c) # Cache the states of current h & c for next iter
        cache = (hf, hi, ho, hc, c, h, y, Wf, Wi, Wc, Wo, Wy, X, c_old) # Add all intermediate variables to this cache

        return prob, state, cache

    def lstm_backward(self, prob, y_train, d_next, cache):
        # Unpack the cache variable to get the intermediate variables used in forward step
        hf, hi, ho, hc, c, h, y, Wf, Wi, Wc, Wo, Wy, X, c_old = cache
        dh_next, dc_next = d_next

        # Softmax loss gradient
        dy = prob.copy()
        dy -= y_train
        dy = sigmoid_derivative(dy)

        # Hidden to output gradient
        dWy = h.T @ dy
        dby = dy
        # Note we're adding dh_next here
        dh = dy @ Wy.T + dh_next

        # Gradient for ho in h = ho * tanh(c)
        dho = tanh(c) * dh
        dho = sigmoid_derivative(ho) * dho

        # Gradient for c in h = ho * tanh(c), note we're adding dc_next here
        dc = ho * dh * tanh_derivative(c)
        dc = dc + dc_next

        # Gradient for hf in c = hf * c_old + hi * hc
        dhf = c_old * dc
        dhf = sigmoid_derivative(hf) * dhf

        # Gradient for hi in c = hf * c_old + hi * hc
        dhi = hc * dc
        dhi = sigmoid_derivative(hi) * dhi

        # Gradient for hc in c = hf * c_old + hi * hc
        dhc = hi * dc
        dhc = tanh_derivative(hc) * dhc

        # Gate gradients, just a normal fully connected layer gradient
        dWf = X.T @ dhf
        dbf = dhf
        dXf = dhf @ Wf.T

        dWi = X.T @ dhi
        dbi = dhi
        dXi = dhi @ Wi.T

        dWo = X.T @ dho
        dbo = dho
        dXo = dho @ Wo.T

        dWc = X.T @ dhc
        dbc = dhc
        dXc = dhc @ Wc.T

        # As X was used in multiple gates, the gradient must be accumulated here
        dX = dXo + dXc + dXi + dXf
        # Split the concatenated X, so that we get our gradient of h_old
        dh_next = dX[:, :self.H]
        # Gradient for c_old in c = hf * c_old + hi * hc
        dc_next = hf * dc

        grad = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)
        state = (dh_next, dc_next)

        return grad, state

    def fit(self, X_train, y_train, state):
        probs = []
        caches = []
        loss = 0.
        h, c = state

        print('Forward Start')

        # Forward Step
        for x, y_true in zip(X_train, y_train):
            prob, state, cache = self.lstm_forward(x, state)
            loss += cross_entropy(prob, y_true)

            # Store forward step result to be used in backward step
            probs.append(prob)
            caches.append(cache)

        print('Forward Finish')
        # The loss is the average cross entropy
        loss /= np.array(X_train).shape[0]

        # Backward Step
        # Gradient for dh_next and dc_next is zero for the last timestep
        d_next = (np.zeros_like(h), np.zeros_like(c))
        grads = {k: np.zeros_like(v) for k, v in self.model.items()}

        # Go backward from the last timestep to the first
        for prob, y_true, cache in reversed(list(zip(probs, y_train, caches))):
            grad, d_next = self.lstm_backward(prob, y_true, d_next, cache)

            # Accumulate gradients from all timesteps
            for k in grads.keys():
                grads[k] += grad[k]

        print('Loss：{}'.format(loss))
        return grads, loss, state


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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        torch.manual_seed(1)
        self.hidden_dim = hidden_dim
        # self.model = nn.ModuleDict({
        #     'word_embeddings': nn.Embedding(vocab_size, embedding_dim),
        #     'lstm': nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim),
        #     'linear': nn.Linear(in_features = hidden_dim, out_features = tagset_size)
        # })
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, X_train, h):
        # Embed word ids to vectors
        x = self.word_embeddings(X_train)
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)

    # def forward(self, X_train, h):
    #     # Embed word ids to vectors
    #     x = self.word_embeddings(X_train)
    #     # Data is fed to the LSTM
    #     out, (h, c) = self.model['lstm'](x)
    #     print(f'lstm output={out.size()}')
    #     # Reshape output to (batch_size*sequence_length, hidden_size)
    #     out = out.reshape(out.size(0)*out.size(1), out.size(2))
    #     # Decode hidden states of all time steps
    #     out = self.linear(out)
    #     return out, (h, c)


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


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words: 
                    self.dictionary.add_word(word)  
        
        # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches*batch_size]
        return ids.view(batch_size, -1)

# 定义函数：截断反向传播
def detach(states):
    return [state.detach() for state in states]