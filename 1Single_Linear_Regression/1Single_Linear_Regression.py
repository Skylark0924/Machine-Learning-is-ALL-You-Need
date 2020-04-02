from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")


class Skylark_LinearRegression():
    def __init__(self, X):
        self.init_theta = np.zeros(X.shape[1])  # X.shape[1]=2,代表特征数n
        self.final_theta = np.zeros(X.shape[1])
        self.epoch = 500
        self.b = 0  # 截距
        self.k = 0  # 斜率
        self.cost = []  # 代价数据

    def fit(self, X, y):
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
        ax = plt.plot()
        self.tsplot(ax, self.cost)
        ax.set_xlabel('epoch')
        ax.set_ylabel('cost')
        plt.show()

    def tsplot(self, ax, data, **kw):
        x = np.arange(data.shape[1])
        est = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        cis = (est - sd, est + sd)
        ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
        ax.plot(x, est, **kw)
        ax.margins(x=0)


if __name__ == '__main__':
    use_sklearn = True

    # Data Preprocessing
    dataset = pd.read_csv('studentscores.csv')
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

    else:
        regressor = Skylark_LinearRegression(X_train)
        regressor.fit(X_train, Y_train)

    # Predecting the Result
    Y_pred = regressor.predict(X_test)

    # Visualization
    # Training Results
    plt.scatter(X_train, Y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    # Testing Results
    plt.scatter(X_test, Y_test, color='red')
    plt.plot(X_test, Y_pred, color='blue')
