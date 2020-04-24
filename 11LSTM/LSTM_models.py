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
sys.path.append("\home\skylark\Github\Machine-Learning-Basic-Codes")
from utils.tool_func import *

class Skylark_LSTM():
    def __init__(self):

    def init_para(self):
        #initialize the parameters with 0 mean and 0.01 standard deviation
        mean = 0
        std = 0.01
        
        #lstm cell weights
        forget_gate_weights = np.random.normal(mean,std,(input_units+hidden_units,hidden_units))
        input_gate_weights  = np.random.normal(mean,std,(input_units+hidden_units,hidden_units))
        output_gate_weights = np.random.normal(mean,std,(input_units+hidden_units,hidden_units))
        gate_gate_weights   = np.random.normal(mean,std,(input_units+hidden_units,hidden_units))
        
        #hidden to output weights (output cell)
        hidden_output_weights = np.random.normal(mean,std,(hidden_units,output_units))
        
        parameters = dict()
        parameters['fgw'] = forget_gate_weights
        parameters['igw'] = input_gate_weights
        parameters['ogw'] = output_gate_weights
        parameters['ggw'] = gate_gate_weights
        parameters['how'] = hidden_output_weights
        
        return parameters

    def lstm_cell(self, batch_dataset, prev_activation_matrix, prev_cell_matrix, parameters):
        '''
        Single lstm cell
        '''
        #get parameters
        fgw = parameters['fgw']
        igw = parameters['igw']
        ogw = parameters['ogw']
        ggw = parameters['ggw']
        
        #concat batch data and prev_activation matrix
        concat_dataset = np.concatenate((batch_dataset,prev_activation_matrix),axis=1)
        
        #forget gate activations
        fa = np.matmul(concat_dataset,fgw)
        fa = sigmoid(fa)
        
        #input gate activations
        ia = np.matmul(concat_dataset,igw)
        ia = sigmoid(ia)
        
        #output gate activations
        oa = np.matmul(concat_dataset,ogw)
        oa = sigmoid(oa)
        
        #gate gate activations
        ga = np.matmul(concat_dataset,ggw)
        ga = tanh_activation(ga)
        
        #new cell memory matrix
        cell_memory_matrix = np.multiply(fa,prev_cell_matrix) + np.multiply(ia,ga)
        
        #current activation matrix
        activation_matrix = np.multiply(oa, tanh_activation(cell_memory_matrix))
        
        #lets store the activations to be used in back prop
        lstm_activations = dict()
        lstm_activations['fa'] = fa
        lstm_activations['ia'] = ia
        lstm_activations['oa'] = oa
        lstm_activations['ga'] = ga
        
        return lstm_activations,cell_memory_matrix,activation_matrix

    def output_cell(self, activation_matrix,parameters):
        #get hidden to output parameters
        how = parameters['how']
        
        #get outputs 
        output_matrix = np.matmul(activation_matrix,how)
        output_matrix = softmax(output_matrix)
        
        return output_matrix

    def get_embeddings(self, batch_dataset,embeddings):
        embedding_dataset = np.matmul(batch_dataset,embeddings)
        return embedding_dataset

    #forward propagation
    def forward_propagation(self, batches,parameters,embeddings):
        #get batch size
        batch_size = batches[0].shape[0]
        
        #to store the activations of all the unrollings.
        lstm_cache = dict()                 #lstm cache
        activation_cache = dict()           #activation cache 
        cell_cache = dict()                 #cell cache
        output_cache = dict()               #output cache
        embedding_cache = dict()            #embedding cache 
        
        #initial activation_matrix(a0) and cell_matrix(c0)
        a0 = np.zeros([batch_size,hidden_units],dtype=np.float32)
        c0 = np.zeros([batch_size,hidden_units],dtype=np.float32)
        
        #store the initial activations in cache
        activation_cache['a0'] = a0
        cell_cache['c0'] = c0
        
        #unroll the names
        for i in range(len(batches)-1):
            #get first first character batch
            batch_dataset = batches[i]
            
            #get embeddings 
            batch_dataset = get_embeddings(batch_dataset,embeddings)
            embedding_cache['emb'+str(i)] = batch_dataset
            
            #lstm cell
            lstm_activations,ct,at = lstm_cell(batch_dataset,a0,c0,parameters)
            
            #output cell
            ot = output_cell(at,parameters)
            
            #store the time 't' activations in caches
            lstm_cache['lstm' + str(i+1)]  = lstm_activations
            activation_cache['a'+str(i+1)] = at
            cell_cache['c' + str(i+1)] = ct
            output_cache['o'+str(i+1)] = ot
            
            #update a0 and c0 to new 'at' and 'ct' for next lstm cell
            a0 = at
            c0 = ct
            
        return embedding_cache,lstm_cache,activation_cache,cell_cache,output_cache

    #calculate loss, perplexity and accuracy
    def cal_loss_accuracy(self, batch_labels,output_cache):
        loss = 0  #to sum loss for each time step
        acc  = 0  #to sum acc for each time step 
        prob = 1  #probability product of each time step predicted char
        
        #batch size
        batch_size = batch_labels[0].shape[0]
        
        #loop through each time step
        for i in range(1,len(output_cache)+1):
            #get true labels and predictions
            labels = batch_labels[i]
            pred = output_cache['o'+str(i)]
            
            prob = np.multiply(prob,np.sum(np.multiply(labels,pred),axis=1).reshape(-1,1))
            loss += np.sum((np.multiply(labels,np.log(pred)) + np.multiply(1-labels,np.log(1-pred))),axis=1).reshape(-1,1)
            acc  += np.array(np.argmax(labels,1)==np.argmax(pred,1),dtype=np.float32).reshape(-1,1)
        
        #calculate perplexity loss and accuracy
        perplexity = np.sum((1/prob)**(1/len(output_cache)))/batch_size
        loss = np.sum(loss)*(-1/batch_size)
        acc  = np.sum(acc)/(batch_size)
        acc = acc/len(output_cache)
        
        return perplexity,loss,acc
    
    #calculate output cell errors
    def calculate_output_cell_error(batch_labels,output_cache,parameters):
        #to store the output errors for each time step
        output_error_cache = dict()
        activation_error_cache = dict()
        how = parameters['how']
        
        #loop through each time step
        for i in range(1,len(output_cache)+1):
            #get true and predicted labels
            labels = batch_labels[i]
            pred = output_cache['o'+str(i)]
            
            #calculate the output_error for time step 't'
            error_output = pred - labels
            
            #calculate the activation error for time step 't'
            error_activation = np.matmul(error_output,how.T)
            
            #store the output and activation error in dict
            output_error_cache['eo'+str(i)] = error_output
            activation_error_cache['ea'+str(i)] = error_activation
            
        return output_error_cache,activation_error_cache

    #calculate error for single lstm cell
    def calculate_single_lstm_cell_error(self, activation_output_error,next_activation_error,next_cell_error,parameters,lstm_activation,cell_activation,prev_cell_activation):
        #activation error =  error coming from output cell and error coming from the next lstm cell
        activation_error = activation_output_error + next_activation_error
        
        #output gate error
        oa = lstm_activation['oa']
        eo = np.multiply(activation_error,tanh_activation(cell_activation))
        eo = np.multiply(np.multiply(eo,oa),1-oa)
        
        #cell activation error
        cell_error = np.multiply(activation_error,oa)
        cell_error = np.multiply(cell_error,tanh_derivative(tanh_activation(cell_activation)))
        #error also coming from next lstm cell 
        cell_error += next_cell_error
        
        #input gate error
        ia = lstm_activation['ia']
        ga = lstm_activation['ga']
        ei = np.multiply(cell_error,ga)
        ei = np.multiply(np.multiply(ei,ia),1-ia)
        
        #gate gate error
        eg = np.multiply(cell_error,ia)
        eg = np.multiply(eg,tanh_derivative(ga))
        
        #forget gate error
        fa = lstm_activation['fa']
        ef = np.multiply(cell_error,prev_cell_activation)
        ef = np.multiply(np.multiply(ef,fa),1-fa)
        
        #prev cell error
        prev_cell_error = np.multiply(cell_error,fa)
        
        #get parameters
        fgw = parameters['fgw']
        igw = parameters['igw']
        ggw = parameters['ggw']
        ogw = parameters['ogw']
        
        #embedding + hidden activation error
        embed_activation_error = np.matmul(ef,fgw.T)
        embed_activation_error += np.matmul(ei,igw.T)
        embed_activation_error += np.matmul(eo,ogw.T)
        embed_activation_error += np.matmul(eg,ggw.T)
        
        input_hidden_units = fgw.shape[0]
        hidden_units = fgw.shape[1]
        input_units = input_hidden_units - hidden_units
        
        #prev activation error
        prev_activation_error = embed_activation_error[:,input_units:]
        
        #input error (embedding error)
        embed_error = embed_activation_error[:,:input_units]
        
        #store lstm error
        lstm_error = dict()
        lstm_error['ef'] = ef
        lstm_error['ei'] = ei
        lstm_error['eo'] = eo
        lstm_error['eg'] = eg
        
        return prev_activation_error,prev_cell_error,embed_error,lstm_error

    #calculate output cell derivatives
    def calculate_output_cell_derivatives(self,output_error_cache,activation_cache,parameters):
        #to store the sum of derivatives from each time step
        dhow = np.zeros(parameters['how'].shape)
        
        batch_size = activation_cache['a1'].shape[0]
        
        #loop through the time steps 
        for i in range(1,len(output_error_cache)+1):
            #get output error
            output_error = output_error_cache['eo' + str(i)]
            
            #get input activation
            activation = activation_cache['a'+str(i)]
            
            #cal derivative and summing up!
            dhow += np.matmul(activation.T,output_error)/batch_size
            
        return dhow

    #calculate derivatives for single lstm cell
    def calculate_single_lstm_cell_derivatives(self, lstm_error,embedding_matrix,activation_matrix):
        #get error for single time step
        ef = lstm_error['ef']
        ei = lstm_error['ei']
        eo = lstm_error['eo']
        eg = lstm_error['eg']
        
        #get input activations for this time step
        concat_matrix = np.concatenate((embedding_matrix,activation_matrix),axis=1)
        
        batch_size = embedding_matrix.shape[0]
        
        #cal derivatives for this time step
        dfgw = np.matmul(concat_matrix.T,ef)/batch_size
        digw = np.matmul(concat_matrix.T,ei)/batch_size
        dogw = np.matmul(concat_matrix.T,eo)/batch_size
        dggw = np.matmul(concat_matrix.T,eg)/batch_size
        
        #store the derivatives for this time step in dict
        derivatives = dict()
        derivatives['dfgw'] = dfgw
        derivatives['digw'] = digw
        derivatives['dogw'] = dogw
        derivatives['dggw'] = dggw
        
        return derivatives

    #backpropagation
    def backward_propagation(self, batch_labels,embedding_cache,lstm_cache,activation_cache,cell_cache,output_cache,parameters):
        #calculate output errors 
        output_error_cache,activation_error_cache = calculate_output_cell_error(batch_labels,output_cache,parameters)
        
        #to store lstm error for each time step
        lstm_error_cache = dict()
        
        #to store embeding errors for each time step
        embedding_error_cache = dict()
        
        # next activation error 
        # next cell error  
        #for last cell will be zero
        eat = np.zeros(activation_error_cache['ea1'].shape)
        ect = np.zeros(activation_error_cache['ea1'].shape)
        
        #calculate all lstm cell errors (going from last time-step to the first time step)
        for i in range(len(lstm_cache),0,-1):
            #calculate the lstm errors for this time step 't'
            pae,pce,ee,le = calculate_single_lstm_cell_error(activation_error_cache['ea'+str(i)],eat,ect,parameters,lstm_cache['lstm'+str(i)],cell_cache['c'+str(i)],cell_cache['c'+str(i-1)])
            
            #store the lstm error in dict
            lstm_error_cache['elstm'+str(i)] = le
            
            #store the embedding error in dict
            embedding_error_cache['eemb'+str(i-1)] = ee
            
            #update the next activation error and next cell error for previous cell
            eat = pae
            ect = pce
        
        
        #calculate output cell derivatives
        derivatives = dict()
        derivatives['dhow'] = calculate_output_cell_derivatives(output_error_cache,activation_cache,parameters)
        
        #calculate lstm cell derivatives for each time step and store in lstm_derivatives dict
        lstm_derivatives = dict()
        for i in range(1,len(lstm_error_cache)+1):
            lstm_derivatives['dlstm'+str(i)] = calculate_single_lstm_cell_derivatives(lstm_error_cache['elstm'+str(i)],embedding_cache['emb'+str(i-1)],activation_cache['a'+str(i-1)])
        
        #initialize the derivatives to zeros 
        derivatives['dfgw'] = np.zeros(parameters['fgw'].shape)
        derivatives['digw'] = np.zeros(parameters['igw'].shape)
        derivatives['dogw'] = np.zeros(parameters['ogw'].shape)
        derivatives['dggw'] = np.zeros(parameters['ggw'].shape)
        
        #sum up the derivatives for each time step
        for i in range(1,len(lstm_error_cache)+1):
            derivatives['dfgw'] += lstm_derivatives['dlstm'+str(i)]['dfgw']
            derivatives['digw'] += lstm_derivatives['dlstm'+str(i)]['digw']
            derivatives['dogw'] += lstm_derivatives['dlstm'+str(i)]['dogw']
            derivatives['dggw'] += lstm_derivatives['dlstm'+str(i)]['dggw']
        
        return derivatives,embedding_error_cache

    #update the parameters using adam optimizer
    #adam optimization
    def update_parameters(self, parameters,derivatives,V,S,t):
        #get derivatives
        dfgw = derivatives['dfgw']
        digw = derivatives['digw']
        dogw = derivatives['dogw']
        dggw = derivatives['dggw']
        dhow = derivatives['dhow']
        
        #get parameters
        fgw = parameters['fgw']
        igw = parameters['igw']
        ogw = parameters['ogw']
        ggw = parameters['ggw']
        how = parameters['how']
        
        #get V parameters
        vfgw = V['vfgw']
        vigw = V['vigw']
        vogw = V['vogw']
        vggw = V['vggw']
        vhow = V['vhow']
        
        #get S parameters
        sfgw = S['sfgw']
        sigw = S['sigw']
        sogw = S['sogw']
        sggw = S['sggw']
        show = S['show']
        
        #calculate the V parameters from V and current derivatives
        vfgw = (beta1*vfgw + (1-beta1)*dfgw)
        vigw = (beta1*vigw + (1-beta1)*digw)
        vogw = (beta1*vogw + (1-beta1)*dogw)
        vggw = (beta1*vggw + (1-beta1)*dggw)
        vhow = (beta1*vhow + (1-beta1)*dhow)
        
        #calculate the S parameters from S and current derivatives
        sfgw = (beta2*sfgw + (1-beta2)*(dfgw**2))
        sigw = (beta2*sigw + (1-beta2)*(digw**2))
        sogw = (beta2*sogw + (1-beta2)*(dogw**2))
        sggw = (beta2*sggw + (1-beta2)*(dggw**2))
        show = (beta2*show + (1-beta2)*(dhow**2))
        
        #update the parameters
        fgw = fgw - learning_rate*((vfgw)/(np.sqrt(sfgw) + 1e-6))
        igw = igw - learning_rate*((vigw)/(np.sqrt(sigw) + 1e-6))
        ogw = ogw - learning_rate*((vogw)/(np.sqrt(sogw) + 1e-6))
        ggw = ggw - learning_rate*((vggw)/(np.sqrt(sggw) + 1e-6))
        how = how - learning_rate*((vhow)/(np.sqrt(show) + 1e-6))
        
        #store the new weights
        parameters['fgw'] = fgw
        parameters['igw'] = igw
        parameters['ogw'] = ogw
        parameters['ggw'] = ggw
        parameters['how'] = how
        
        #store the new V parameters
        V['vfgw'] = vfgw 
        V['vigw'] = vigw 
        V['vogw'] = vogw 
        V['vggw'] = vggw
        V['vhow'] = vhow
        
        #store the s parameters
        S['sfgw'] = sfgw 
        S['sigw'] = sigw 
        S['sogw'] = sogw 
        S['sggw'] = sggw
        S['show'] = show
        
        return parameters,V,S

    #update the Embeddings
    def update_embeddings(self, embeddings,embedding_error_cache,batch_labels):
        #to store the embeddings derivatives
        embedding_derivatives = np.zeros(embeddings.shape)
        
        batch_size = batch_labels[0].shape[0]
        
        #sum the embedding derivatives for each time step
        for i in range(len(embedding_error_cache)):
            embedding_derivatives += np.matmul(batch_labels[i].T,embedding_error_cache['eemb'+str(i)])/batch_size
        
        #update the embeddings
        embeddings = embeddings - learning_rate*embedding_derivatives
        return embeddings    

    def initialize_V(self, parameters):
        Vfgw = np.zeros(parameters['fgw'].shape)
        Vigw = np.zeros(parameters['igw'].shape)
        Vogw = np.zeros(parameters['ogw'].shape)
        Vggw = np.zeros(parameters['ggw'].shape)
        Vhow = np.zeros(parameters['how'].shape)
        
        V = dict()
        V['vfgw'] = Vfgw
        V['vigw'] = Vigw
        V['vogw'] = Vogw
        V['vggw'] = Vggw
        V['vhow'] = Vhow
        return V

    def initialize_S(self, parameters):
        Sfgw = np.zeros(parameters['fgw'].shape)
        Sigw = np.zeros(parameters['igw'].shape)
        Sogw = np.zeros(parameters['ogw'].shape)
        Sggw = np.zeros(parameters['ggw'].shape)
        Show = np.zeros(parameters['how'].shape)
        
        S = dict()
        S['sfgw'] = Sfgw
        S['sigw'] = Sigw
        S['sogw'] = Sogw
        S['sggw'] = Sggw
        S['show'] = Show
        return S

    #train function
    def train(self, train_dataset,iters=1000,batch_size=20):
        #initalize the parameters
        parameters = initialize_parameters()
        
        #initialize the V and S parameters for Adam
        V = initialize_V(parameters)
        S = initialize_S(parameters)
        
        #generate the random embeddings
        embeddings = np.random.normal(0,0.01,(len(vocab),input_units))
        
        #to store the Loss, Perplexity and Accuracy for each batch
        J = []
        P = []
        A = []
        
        
        for step in range(iters):
            #get batch dataset
            index = step%len(train_dataset)
            batches = train_dataset[index]
            
            #forward propagation
            embedding_cache,lstm_cache,activation_cache,cell_cache,output_cache = forward_propagation(batches,parameters,embeddings)
            
            #calculate the loss, perplexity and accuracy
            perplexity,loss,acc = cal_loss_accuracy(batches,output_cache)
            
            #backward propagation
            derivatives,embedding_error_cache = backward_propagation(batches,embedding_cache,lstm_cache,activation_cache,cell_cache,output_cache,parameters)
            
            #update the parameters
            parameters,V,S = update_parameters(parameters,derivatives,V,S,step)
            
            #update the embeddings
            embeddings = update_embeddings(embeddings,embedding_error_cache,batches)
            
            
            J.append(loss)
            P.append(perplexity)
            A.append(acc)
            
            #print loss, accuracy and perplexity
            if(step%1000==0):
                print("For Single Batch :")
                print('Step       = {}'.format(step))
                print('Loss       = {}'.format(round(loss,2)))
                print('Perplexity = {}'.format(round(perplexity,2)))
                print('Accuracy   = {}'.format(round(acc*100,2)))
                print()
        
        return embeddings, parameters,J,P,A




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