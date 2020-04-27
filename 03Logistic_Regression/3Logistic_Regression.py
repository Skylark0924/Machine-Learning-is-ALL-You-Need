import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("D:\Github\Machine-Learning-Basic-Codes")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.tool_func import *
from utils.visualize import *


class Skylark_LogisticRegression():
    def __init__(self, learning_rate=0.1, epoch=500):
        super().__init__()
        self.learning_rate = learning_rate
        self.epoch = epoch

        self.init_theta = None  
        self.final_theta = None

    def initialize_weights(self, n_features):
        # 初始化参数
        # 参数范围[-1/sqrt(N), 1/sqrt(N)]
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.init_theta = np.insert(w, 0, b, axis=0)

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.initialize_weights(n_features)
        # 为X增加一列特征x1，x1 = 0
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))

        # 梯度训练n_iterations轮
        for i in range(self.epoch):
            h_x = X.dot(self.init_theta)
            y_pred = sigmoid(h_x)
            theta_grad = X.T.dot(y_pred - y)
            self.init_theta -= self.learning_rate * theta_grad
        self.final_theta = self.init_theta

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        h_x = X.dot(self.final_theta)
        y_pred = np.round(sigmoid(h_x))
        return y_pred.astype(int)

    def cost(self, theta, X, y):
        ''' cost fn is -l(theta) for you to minimize'''
        return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

    def gradient(self, theta, X, y):
        '''just 1 batch gradient'''
        return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)


if __name__ == '__main__':
    use_sklearn = False

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

    if use_sklearn:
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(solver='lbfgs')
        classifier.fit(X_train, Y_train)
    else:
        classifier = Skylark_LogisticRegression()
        classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    print_confusion_matrix(
        Y_test, Y_pred, clf_name='Logistic Regression')

    # Visualising the Training set results
    visualization_clf(X_train, Y_train, classifier,
                      clf_name='Logistic Regression', set_name='Training')
    # Visualising the Test set results
    visualization_clf(X_test, Y_test, classifier,
                      clf_name='Logistic Regression', set_name='Test')
