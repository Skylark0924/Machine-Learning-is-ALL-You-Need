import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # 隐藏warning

from torch import nn
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
            Activation('softmax'),
        ])
        self.classifier.summary()

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
    
    def evaluate(self, X_test, Y_test):
        scores = self.classifier.evaluate(X_test, Y_test, verbose=1)
        # Visualize the result
        print('Test loss: {}\nTest accuracy: {}'.format(scores[0], scores[1]))

class Torch_CNN(nn.Module):
    def __init__(self, num_classes):
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
            nn.Linear(84, num_classes)
        ).to(self.device)
        print('Model:\n{}\nDevice: {}'.format(self.model, self.device))
    
    def fit(self, trainloader, epochs, batch_size, learning_rate=0.001):
       # Define a Loss func & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

        # Training
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
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

    def evaluate(self, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

class TF_CNN():
    def __init__(self):
        super().__init__()
    
    def conv_net(self, x, keep_prob):
        conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
        conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
        conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
        conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

        # 1, 2
        conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
        conv1 = tf.nn.relu(conv1)
        conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        conv1_bn = tf.layers.batch_normalization(conv1_pool)

        # 3, 4
        conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
        conv2 = tf.nn.relu(conv2)
        conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
        conv2_bn = tf.layers.batch_normalization(conv2_pool)
    
        # 5, 6
        conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
        conv3 = tf.nn.relu(conv3)
        conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
        conv3_bn = tf.layers.batch_normalization(conv3_pool)
        
        # 7, 8
        conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
        conv4 = tf.nn.relu(conv4)
        conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        conv4_bn = tf.layers.batch_normalization(conv4_pool)
        
        # 9
        flat = tf.contrib.layers.flatten(conv4_bn)  

        # 10
        full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
        full1 = tf.nn.dropout(full1, keep_prob)
        full1 = tf.layers.batch_normalization(full1)
        
        # 11
        full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
        full2 = tf.nn.dropout(full2, keep_prob)
        full2 = tf.layers.batch_normalization(full2)
        
        # 12
        full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
        full3 = tf.nn.dropout(full3, keep_prob)
        full3 = tf.layers.batch_normalization(full3)    
        
        # 13
        full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
        full4 = tf.nn.dropout(full4, keep_prob)
        full4 = tf.layers.batch_normalization(full4)        
        
        # 14
        out = tf.contrib.layers.fully_connected(inputs=full4, num_outputs=10, activation_fn=None)
        return out

    def fit(self, X_train, Y_train, epochs, batch_size, learning_rate=0.0001):
        # Remove previous weights, bias, inputs, etc..
        tf.reset_default_graph()
        keep_probability = 0.7
        # Inputs
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
        y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        logits = self.conv_net(X_train, keep_prob)

        # Loss and Optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        
        with tf.Session() as sess:
            # Initializing the variables
            sess.run(tf.global_variables_initializer())
            
            # Training cycle
            for epoch in range(epochs):
                # Loop over all batches
                n_batches = 5
                for batch_i in range(1, n_batches + 1):
                    for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                        self.train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                            
                        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                        self.print_stats(sess, batch_features, batch_labels, cost, accuracy)
    
    def train_neural_network(self, session, optimizer, keep_probability, feature_batch, label_batch):
        session.run(optimizer, 
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: keep_probability
                    })

    def print_stats(self, sess, feature_batch, label_batch, cost, accuracy):
        loss = sess.run(cost, 
                        feed_dict={
                            x: feature_batch,
                            y: label_batch,
                            keep_prob: 1.
                        })
        valid_acc = sess.run(accuracy, 
                            feed_dict={
                                x: valid_features,
                                y: valid_labels,
                                keep_prob: 1.
                            })
        
        print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))


def keras_data():
    # Data Preprocessing
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    # Convert class vectors to binary class matrices.
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    input_size = X_train.shape[1:]
    return X_train, Y_train, X_test, Y_test

def Torch_data():
    transform = transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)
    return trainloader, testloader

def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded