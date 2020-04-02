import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Skylark_LinearRegression():
    def __init__(self):
        self.init_theta = None
        self.final_theta = None
        self.epoch = 500
        self.b = 0  # 截距
        self.k = 0  # 斜率
        self.cost = []  # 代价数据

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
        final_theta, cost_data = self.batch_gradient_decent(
            self.init_theta, X, y, self.epoch)
        self.final_theta = final_theta
        self.b = self.final_theta[0]
        self.k = self.final_theta[1]
        self.cost = cost_data

    def predict(self, X):
        Y = X*self.k+self.b
        return Y

    def batch_gradient_decent(self, theta, X, y, epoch, alpha=0.01):
        '''
        批量梯度下降, 拟合线性回归, 返回参数和代价
        epoch: 批处理的轮数
        theta: 网络参数
        alpha: 学习率
        '''
        cost_data = [self.lr_cost(theta, X, y)]
        _theta = theta.copy()  # 拷贝一份，不和原来的theta混淆

        for _ in range(epoch):
            _theta = _theta - alpha * self.gradient(_theta, X, y)
            cost_data.append(self.lr_cost(_theta, X, y))

        return _theta, cost_data

    def gradient(self, theta, X, y):
        m = X.shape[0]
        # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)
        inner = X.T @ (X @ theta - y)
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
        cost = square_sum / (2 * m)

        return cost

    def visual_cost(self):
        print(self.cost)
        figure, ax = plt.subplots()
        nums = np.arange(len(self.cost))
        ax.plot(nums, np.array(self.cost).reshape((len(self.cost,))))
        ax.set_xlabel('epoch')
        ax.set_ylabel('cost')
        plt.show()


if __name__ == '__main__':
    use_sklearn = True

    # Data Preprocessing
    dataset = pd.read_csv('50_Startups.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:,  4].values

    # Encoding Categorical data
    labelencoder = LabelEncoder()
    X[:, 3] = labelencoder.fit_transform(X[:, 3])
    onehotencoder = OneHotEncoder(categorical_features=[3])
    X = onehotencoder.fit_transform(X).toarray()

    # Avoiding Dummy Variable Trap
    X = X[:, 1:]

    # Making Dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)

    if use_sklearn:
        from sklearn.linear_model import LinearRegression

        # Fitting Simple Linear Regression Model to the training set
        regressor = LinearRegression()
        regressor.fit(X_train, Y_train)

    else:
        regressor = Skylark_LinearRegression()
        regressor.fit(X_train, Y_train)
        regressor.visual_cost()

    # Predecting the Result
    Y_pred = regressor.predict(X_test)

    # Visualization
    # Training Results
    plt.scatter(X_train, Y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    # Testing Results
    plt.scatter(X_test, Y_test, color='red')
    plt.plot(X_test, Y_pred, color='blue')
