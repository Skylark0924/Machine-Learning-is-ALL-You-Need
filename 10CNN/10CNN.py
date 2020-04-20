from torch import nn
import torch
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("D:\Github\Machine-Learning-Basic-Codes")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.visualize import *
from utils.tool_func import *

class Skylark_Neural_Network():
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

class Torch_NN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        # define model architecture
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('gpu')
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_classes),
            nn.Sigmoid()
        ).to(self.device)
        
        print('Model:\n{}\nDevice: {}'.format(self.model, self.device))
    #     self.fc1 = nn.Linear(input_size, hidden_size[0]) 
    #     self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
    #     self.fc2 = nn.Linear(hidden_size[1], num_classes)
    #     
    
    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = nn.ReLU(x)
    #     x = self.fc2(x)
    #     x = nn.ReLU(x)
    #     x = self.fc3(x)
    #     out=nn.Sigmoid(x)
    #     return out
    
    def fit(self, X_train, Y_train, epochs, batch_size, learning_rate):
        dtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        tensor_x = torch.Tensor(X_train) # transform from array to torch tensor
        tensor_y = torch.Tensor(Y_train)
        MyDataset = data.TensorDataset(tensor_x, tensor_y) # making the dataset
        # 数据加载器 DataLoader
        # 训练数据加载器
        train_loader = data.DataLoader(
            dataset=MyDataset, 
            batch_size=batch_size, shuffle=True)
        total_step = len(train_loader)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            for i, (x, labels) in enumerate(train_loader):
                x = x.to(self.device, dtype= torch.float)
                labels = labels.to(self.device, dtype= torch.long)

                # 前向传播
                outputs = self.model(x)
                loss = criterion(outputs, labels)

                # 反向传播并优化
                optimizer.zero_grad()  # 注意每步迭代都需要清空梯度缓存
                loss.backward()
                optimizer.step()

                if (i+1) % 30 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch+1, epochs, i+1, total_step, loss.item()))

    def predict(self, X_test):
        X_test = torch.Tensor(X_test).to(self.device)
        Y_pred = self.model(X_test)
        _, Y_pred = torch.max(Y_pred.data, 1)
        return Y_pred.cpu().detach().numpy()

if __name__ == '__main__':
    mode = 'self_implement'  # ['use_sklearn', 'use_keras', 'use_torch', 'self_implement']
    input_size = 2
    hidden_sizes = [12, 8]
    num_classes = 2
    output_size = 1
    learning_rate = 1e-5

    # Data Preprocessing
    dataset = pd.read_csv('./dataset/Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    Y = dataset.iloc[:, 4].values

    # Making Dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train.astype(np.float64))
    X_test = sc.transform(X_test.astype(np.float64))

    if mode == 'use_sklearn':
        from sklearn.neural_network import MLPClassifier
        classifier = MLPClassifier(
            solver='lbfgs', alpha=learning_rate, hidden_layer_sizes=hidden_sizes, random_state=1)
        classifier.fit(X_train, Y_train)
    elif mode == 'use_keras':
        from keras.models import Sequential
        from keras.layers import Dense
        # define the keras model
        classifier = Sequential()
        classifier.add(Dense(hidden_sizes[0], input_dim=input_size, activation='relu'))
        classifier.add(Dense(hidden_sizes[1], activation='relu'))
        classifier.add(Dense(output_size, activation='sigmoid'))
        # compile the keras model
        classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        classifier.fit(X_train, Y_train, epochs=150, batch_size=10)
    elif mode == 'use_torch':
        classifier = Torch_NN(input_size, hidden_sizes, num_classes)
        classifier.fit(X_train, Y_train, epochs=150, batch_size=10, learning_rate = learning_rate)        
    elif mode == 'self_implement': # self-implement
        classifier = Skylark_Neural_Network(input_size, hidden_sizes, num_classes)
        classifier.fit(X_train, Y_train, epochs=100, batch_size=10, learning_rate = learning_rate)
    else:
        print('Attention: Wrong Mode!')

    Y_pred = classifier.predict(X_test)

    # # Making the Confusion Matrix
    # print_confusion_matrix(
    #     Y_test, Y_pred, clf_name='MLP Classification')

    # Visualising the Training set results
    visualization_clf(X_train, Y_train, classifier,
                      clf_name='MLP Classification', set_name='Training')
    # Visualising the Test set results
    visualization_clf(X_test, Y_test, classifier,
                      clf_name='MLP Classification', set_name='Test')
