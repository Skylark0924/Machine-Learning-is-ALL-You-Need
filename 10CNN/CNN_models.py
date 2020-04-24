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

from conv2d import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

class Skylark_CNN():
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv2d = Conv3x3(8)                # 32x32x1 -> 30x30x8
        self.pool = MaxPool2()                  # 30x30x8 -> 15x15x8
        self.softmax = Softmax(15 * 15 * 8, 10) # 15x15x8 -> 10
    
    def fit(self, X_train, Y_train, epochs, batch_size, learning_rate):
        for i in range(epochs): # trains the CNN in epochs
            loss = 0
            num_correct = 0
            for j, (image, label) in enumerate(zip(X_train, Y_train)):
                if j % 100 == 99:
                    print(
                        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                        (j + 1, loss / 100, num_correct))
                    loss = 0
                    num_correct = 0

                loss, acc = self.train(image, label, lr=learning_rate)
                loss += loss
                num_correct += acc
    
    def forward(self, image, label):
        out = self.conv2d.forward((image/255)-0.5)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)

        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0
        return out, loss, acc

    def train(self, image, label, lr = 0.005):
        out, loss, acc = self.forward(image, label)

        # Calculate initial gradient
        gradient = np.zeros(self.num_classes)
        gradient[label] = -1 / out[label]

        # Backprop
        gradient = self.softmax.backprop(gradient, lr)
        gradient = self.pool.backprop(gradient)
        gradient = self.conv2d.backprop(gradient, lr)
        return loss, acc

    def predict(self, X_test):
        out = self.conv2d.forward((X_test/255)-0.5)
        out = self.pool.forward(out)
        y_pred = self.softmax.forward(out)
        return y_pred
    
    def evaluate(self, X_test, Y_test):
        num_correct = 0
        total_loss = 0
        for j, (image, label) in enumerate(zip(X_test, Y_test)):
            _, loss, acc = self.forward(image, label)
            num_correct += acc
            total_loss += loss
        print('Test loss: {}\nTest accuracy: {}'.format(total_loss/X_test.shape[0], num_correct/X_test.shape[0]))

class Keras_CNN():
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        self.classifier = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=input_size), # https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/
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
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.X = tf.placeholder(tf.float32, [None, self.input_size])
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                            padding='SAME')

    #创建模型
    def conv_net(self, x, weights, biases, dropout):
        x = tf.reshape(x, shape=[-1, 32, 32, 1])
        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)
        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    def init_para(self, num_classes):
        # 设置权重和偏移
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), ### 32
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([8*8*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, num_classes]))
        }

        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }
        return weights, biases

    def fit(self, X_train, Y_train, sess, epochs, dropout=0.75, batch_size=128, learning_rate=0.001):
        weights, biases = self.init_para(self.num_classes)
        X_train = X_train.reshape((X_train.shape[0], self.input_size))
        # Construct model
        logits = self.conv_net(self.X, weights, biases, self.keep_prob)
        prediction = tf.nn.softmax(logits)
        pred = tf.argmax(prediction, 1)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Evaluate model
        self.correct_pred = tf.equal(pred, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.init = tf.global_variables_initializer()
        # saver = tf.train.Saver(tf.trainable_variables())
        
        self.train(X_train, Y_train, sess, epochs, dropout, batch_size, learning_rate)

    def train(self, X_train, Y_train, sess, epochs, dropout=0.75, batch_size=128, learning_rate=0.001):
        display_step = 50 #显示间隔
        sess.run(self.init)
        for epoch in range(1, epochs+1):
            batch_x, batch_y = X_train[batch_size*(epoch-1): batch_size*epoch], Y_train[batch_size*(epoch-1): batch_size*epoch]
            # Run optimization op (backprop)
            sess.run(self.train_op, feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: dropout})
            if epoch % display_step == 0 or epoch == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={self.X: batch_x,
                                                                    self.Y: batch_y,
                                                                    self.keep_prob: 1.0})
                print("Step " + str(epoch) + ", Minibatch Loss={:.4f}".format(loss) + ", Training Accuracy={:.3f}".format(acc))

        
    def evaluate(self, X_test, Y_test, sess):
        X_test = X_test.reshape((X_test.shape[0], self.input_size))
        print('Test Acc: {}'.format(sess.run(self.accuracy, feed_dict={self.X: X_test[:500],
                                        self.Y: Y_test[:500],
                                        self.keep_prob: 1.0})))



def keras_data(num_classes):
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

def rgb2gray(rgb):
    """Convert from color image (RGB) to grayscale.
       Source: opencv.org
       grayscale = 0.299*red + 0.587*green + 0.114*blue
    Argument:
        rgb (tensor): rgb image
    Return:
        (tensor): grayscale image
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])