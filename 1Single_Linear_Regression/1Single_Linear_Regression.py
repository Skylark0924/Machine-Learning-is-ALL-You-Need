from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import sys
sys.path.append("D:\Github\Machine-Learning-Basic-Codes")
import warnings
warnings.filterwarnings('ignore')

from utils.tool_func import *
from utils.visualize import *

class Skylark_LinearRegression():
    def __init__(self, n_epoch=500, learning_rate=0.01, regularization=None, use_gradient=True):
        self.epoch = n_epoch
        self.learning_rate = learning_rate
        self.use_gradient = use_gradient  # 是否使用梯度下降法

        self.init_theta = None  # 初始化参数
        self.final_theta = None  # 最终参数
        self.cost = []  # 代价数据

        if regularization == None:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0
        else:
            self.regularization = regularization

    def initialize_weights(self, n_features):
        # 随机初始化参数
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.init_theta = np.insert(w, 0, b, axis=0)

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.initialize_weights(n_features)
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))
        if self.use_gradient == True:
            # 使用梯度下降法
            final_theta, cost_data = self.batch_gradient_decent(
                self.init_theta, X, y, self.epoch, self.learning_rate)
            self.final_theta = final_theta
            self.cost = cost_data
        else:
            # 使用正规方程法
            X = np.matrix(X)
            y = np.matrix(y)
            X_T_X = X.T.dot(X)
            X_T_X_I_X_T = X_T_X.I.dot(X.T)
            X_T_X_I_X_T_X_T_y = X_T_X_I_X_T.dot(y)
            self.final_theta = X_T_X_I_X_T_X_T_y

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.final_theta)
        return y_pred

    def batch_gradient_decent(self, theta, X, y, epoch, learning_rate):
        '''
        批量梯度下降, 拟合线性回归, 返回参数和代价
        epoch: 批处理的轮数
        theta: 网络参数
        learning_rate: 学习率
        '''
        cost_data = [self.lr_cost(theta, X, y)]
        _theta = theta.copy()  # 拷贝一份，不和原来的theta混淆

        for _ in range(epoch):
            _theta = _theta - learning_rate * self.gradient(_theta, X, y)
            cost_data.append(self.lr_cost(_theta, X, y))

        return _theta, cost_data

    def gradient(self, theta, X, y):
        m = X.shape[0]
        # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)
        inner = X.T @ (X @ theta - y) + self.regularization.grad(theta)
        return inner / m

    def lr_cost(self, theta, X, y):
        '''
        X: R(m*n), m 样本数, n 特征数
        y: R(m)
        theta : R(n), 线性回归的参数
        '''
        m = X.shape[0]  # m为样本数

        inner = X @ theta - y  # R(m*1)，X @ theta等价于X.dot(theta)

        # 1*m @ m*1 = 1*1 in matrix multiplication
        # but you know numpy didn't do transpose in 1d array, so here is just a
        # vector inner product to itselves
        square_sum = inner.T @ inner
        cost = square_sum / (2 * m) + self.regularization(theta)

        return cost

    def visual_cost(self):
        figure, ax = plt.subplots()
        nums = np.arange(len(self.cost))
        ax.plot(nums, np.array(self.cost).reshape((len(self.cost,))))
        ax.set_xlabel('epoch')
        ax.set_ylabel('cost')
        plt.show()


if __name__ == '__main__':
    use_sklearn = False

    # Data Preprocessing
    dataset = pd.read_csv('./dataset/studentscores.csv')
    X = dataset.iloc[:, : 1].values
    Y = dataset.iloc[:, 1].values

    # Making Dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1/4, random_state=0)

    if use_sklearn:
        from sklearn.linear_model import LinearRegression
        # Fitting Simple Linear Regression Model to the training set
        regressor = LinearRegression()
        regressor = regressor.fit(X_train, Y_train)

    else:  # 使用自定义类
        regressor = Skylark_LinearRegression()
        regressor.fit(X_train, Y_train)
        # regressor.visual_cost()

    # Predecting the Result
    Y_pred = regressor.predict(X_test)

    # MSE
    print_mse(Y_test, Y_pred, reg_name='SLR')

    # Visualization
    visualization_reg(X_train, Y_train, regressor,
                  reg_name='Single Linear Regression', set_name='Training')
    visualization_reg(X_test, Y_test, regressor,
                  reg_name='Single Linear Regression', set_name='Test')
