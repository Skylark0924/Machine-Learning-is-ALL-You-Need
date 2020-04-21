import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from torch import nn
import torch
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import sys
# sys.path.append("D:\Github\Machine-Learning-Basic-Codes")

# from utils.visualize import *
# from utils.tool_func import *

class Skylark_CNN():
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        # y=wx+b
        self.w1 = np.random.rand(hidden_sizes[0], input_size)
        self.w2 = np.random.rand(hidden_sizes[1], hidden_sizes[0])
        self.w3 = np.random.rand(num_classes, hidden_sizes[1])
    
    def fit(self, X_train, Y_train, epochs, batch_size, learning_rate):
        self.X = X_train.T
        self.Y = Y_train.T
        self.learning_rate = learning_rate
        for i in range(epochs): # trains the NN 1,000 times
            if i % 50 ==0: 
                print ("For iteration # " + str(i) + "\n")
                print ("Loss: \n" + str(np.mean(np.square(Y_train - self.feedforward())))) # mean sum squared loss
                print ("\n")
            self.output = self.feedforward()
            self.backprop()
    
    def feedforward(self):
        self.layer1 = relu(np.dot(self.w1, self.X))
        self.layer2 = relu(np.dot(self.w2, self.layer1))
        self.layer3 = sigmoid(np.dot(self.w3, self.layer2))
        return self.layer3

    def backprop(self):
        input_data = self.X
        temp3 = 2*(self.Y - self.output) * sigmoid_derivative(self.output)
        d_w3 = np.dot(temp3, self.layer2.T)
        temp2 = np.dot(self.w3.T, temp3) * relu_derivative(self.layer2)
        d_w2 = np.dot(temp2, self.layer1.T)
        temp1 = np.dot(self.w2.T, temp2) * relu_derivative(self.layer1)
        d_w1 = np.dot(temp1, input_data.T)

        # Update parameters
        self.w1 += self.learning_rate * d_w1
        self.w2 += self.learning_rate * d_w2   
        self.w3 += self.learning_rate * d_w3 

    def predict(self, X_test):
        self.X = X_test.T
        y_pred = self.feedforward()
        return np.argmax(np.array(y_pred).T, axis=1).T

class Keras_CNN():
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        self.classifier = Sequential([
            Conv2D(32, (3, 3), padding='same', input_shape=input_size),
            Activation('relu'),
            Conv2D(32, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Conv2D(64, (3, 3), padding='same'),
            Activation('relu'),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(512),
            Activation('relu'),
            Dropout(0.5),
            Dense(num_classes),
            Activation('softmax')
        ])

    def fit(self, X_train, Y_train, epochs, batch_size, learning_rate=0.0001):
        X_train = X_train.astype('float32')
        X_train /= 255
        # initiate RMSprop optimizer
        opt = keras.optimizers.RMSprop(lr=learning_rate, decay=1e-6)
        # Let's train the model using RMSprop
        self.classifier.compile(loss='categorical_crossentropy', optimizer=opt,
                    metrics=['accuracy'])
        self.classifier.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True)

    def predict(self, X_test):
        y_pred = self.classifier.predict(X_test)
        return np.array(y_pred)

class Torch_CNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # define model architecture
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        ).to(self.device)
        print('Model:\n{}\nDevice: {}'.format(self.model, self.device))
    
    def fit(self, X_train, Y_train, epochs, batch_size, learning_rate=0.001):
        # prepare the dataloader
        dtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        tensor_x = torch.Tensor(X_train) # transform from array to torch tensor
        tensor_y = torch.Tensor(Y_train)
        MyDataset = data.TensorDataset(tensor_x, tensor_y) # making the dataset
        train_loader = data.DataLoader(
            dataset=MyDataset, 
            batch_size=batch_size, shuffle=True)
        
        # Define a Loss func & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

        # Training
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device, dtype= torch.float)
                labels = labels.to(self.device, dtype= torch.long)

                # Forward
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward + Optimize
                optimizer.zero_grad()  # 注意每步迭代都需要清空梯度缓存
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training')

    def predict(self, X_test):
        X_test = torch.Tensor(X_test).to(self.device)
        Y_pred = self.model(X_test)
        _, Y_pred = torch.max(Y_pred.data, 1)
        return Y_pred.cpu().detach().numpy()

if __name__ == '__main__':
    mode = 'use_keras'  # ['use_tf', 'use_keras', 'use_torch', 'self_implement']
    
    hidden_sizes = [12, 8]
    num_classes = 10
    output_size = 1
    learning_rate = 0.0001

    # Data Preprocessing
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    # Convert class vectors to binary class matrices.
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    input_size = X_train.shape[1:]

    if mode == 'use_keras':
        classifier = Keras_CNN(input_size, hidden_sizes, num_classes)
        classifier.fit(X_train, Y_train, epochs=100, batch_size=32, learning_rate = learning_rate)
    elif mode == 'use_torch':
        classifier = Torch_CNN(input_size, hidden_sizes, num_classes)
        classifier.fit(X_train, Y_train, epochs=150, batch_size=10, learning_rate = learning_rate)        
    elif mode == 'use_tf':
        classifier = MLPClassifier(
            solver='lbfgs', alpha=learning_rate, hidden_layer_sizes=hidden_sizes, random_state=1)
        classifier.fit(X_train, Y_train)
    elif mode == 'self_implement': # self-implement
        classifier = Skylark_CNN(input_size, hidden_sizes, num_classes)
        classifier.fit(X_train, Y_train, epochs=100, batch_size=10, learning_rate = learning_rate)
    else:
        print('Attention: Wrong Mode!')

    Y_pred = classifier.predict(X_test)

