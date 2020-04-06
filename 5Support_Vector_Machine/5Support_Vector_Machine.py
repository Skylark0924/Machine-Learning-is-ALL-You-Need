import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import progressbar

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kernels import *

class Skylark_SVC():
    def __init__(self, trainX,trainY, C=1, kernel=None, difference=1e-3, max_iter=100):
        super().__init__()
        self.C = C  #正则化的参数
        self.difference = difference #用来判断是否收敛的阈值
        self.max_iter = max_iter #迭代次数的最大值
        self.b = 0 # 偏置值
        self.alpha = None # 拉格朗日乘子
        self.K = None # 特征经过核函数转化的值
        self.X = trainX
        self.Y = trainY
        self.m = trainX.shape[0]
        self.n = trainX.shape[1]
        self.K = np.zeros((self.m, self.m)) #核的新特征数组初始化
        # self.bar = progressbar.ProgressBar(widgets=bar_widgets)  # 进度条
        if kernel == None:
            self.kernel = LinearKernel()  # 无核默认是线性的核
        else:
            self.kernel = kernel
        for i in range(self.m):
            self.K[:, i] = self.kernel(self.X, self.X[i, :]) #每一行数据的特征通过核函数转化 n->m

        self.alpha = np.zeros(self.m) #拉格朗日乘子初始化



    def fit(self):
        for now_iter in range(self.max_iter):

            alpha_prev = np.copy(self.alpha)
            for j in range(self.m):

                #选择第二个优化的拉格朗日乘子
                i = self.random_index(j)
                error_i, error_j = self.error_row(i), self.error_row(j)

                #检验他们是否满足KKT条件，然后选择违反KKT条件最严重的self.alpha[j]
                if (self.Y[j] * error_j < -0.001 and self.alpha[j] < self.C) or (self.Y[j] * error_j > 0.001 and self.alpha[j] > 0):

                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]  #第j个要优化的拉格朗日乘子，最后需要的

                    if eta >= 0:
                        continue

                    L, H = self.getBounds(i, j)
                    old_alpha_j, old_alpha_i = self.alpha[j], self.alpha[i]  #旧的拉格朗日乘子的值
                    self.alpha[j] -= (self.Y[j] * (error_i - error_j)) / eta  #self.alpha[j]的更新

                    #根据约束最后更新拉格朗日乘子self.alpha[j]，并且更新self.alpha[j]
                    self.alpha[j] = self.finalValue(self.alpha[j], H, L)
                    self.alpha[i] = self.alpha[i] + self.Y[i] * self.Y[j] * (old_alpha_j - self.alpha[j])

                    #更新偏置值b
                    b1 = self.b - error_i - self.Y[i] * (self.alpha[i] - old_alpha_j) * self.K[i, i] - \
                         self.Y[j] * (self.alpha[j] - old_alpha_j) * self.K[i, j]
                    b2 = self.b - error_j - self.Y[j] * (self.alpha[j] - old_alpha_j) * self.K[j, j] - \
                         self.Y[i] * (self.alpha[i] - old_alpha_i) * self.K[i, j]
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)

            #判断是否收敛
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.difference:
                break

    def predict(self, X):
        n = X.shape[0]
        result = np.zeros(n)
        for i in range(n):
            result[i] = np.sign(self.predict_row(X[i, :])) #正的返回1，负的返回-1
        return result


    #随机一个要优化的拉格朗日乘子，该乘子必须和循环里面选择的乘子不同
    def random_index(self, first_alpha):
        i = first_alpha
        while i == first_alpha:
          i = np.random.randint(0, self.m - 1)
        return i

    # def kernelTrans(self, ):
    #     ...

    #用带拉格朗日乘子表示的w代入wx+b
    def predict_row(self, X):
        k_v = self.kernel(self.X, X)

        return np.dot((self.alpha * self.Y).T, k_v.T) + self.b

    #预测的值减真实的Y
    def error_row(self, i):

        return self.predict_row(self.X[i]) - self.Y[i]

    #得到self.alpha[j]的范围约束
    def getBounds(self,i,j):

        if self.Y[i] != self.Y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H


    #根据self.alpha[i]的范围约束获得最终的值
    def finalValue(self,alpha,H,L):

        if alpha > H:
            alpha = H
        elif alpha < L:
            alpha = L

        return alpha


if __name__ == '__main__':
    use_sklearn = True

    # Data Preprocessing
    dataset = pd.read_csv('./dataset/Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    Y = dataset.iloc[:, 4].values

    # Making Dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    if use_sklearn:
        from sklearn.svm import SVC
        classifier = SVC(kernel='linear', random_state=0)
        classifier.fit(X_train, Y_train)
    else:
        classifier = Skylark_SVC(X_train, Y_train)
        classifier.fit()

    Y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)

    # Visualization
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, Y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('SVM (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
