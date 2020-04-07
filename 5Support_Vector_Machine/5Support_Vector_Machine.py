import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
sys.path.append("D:\Github\Machine-Learning-Basic-Codes") 
# import progressbar

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kernels import *
from utils.visualize import *


class Skylark_SVC():
    def __init__(self, C=1, kernel=None, epsilon=1e-3, epoch=600):
        super().__init__()
        self.C = C  # 正则化的参数
        self.kernel = kernel
        self.epsilon = epsilon  # 用来判断是否收敛的阈值
        self.epoch = epoch  # 迭代次数的最大值
        self.b = 0  # 偏置值
        self.alpha = None  # 拉格朗日乘子
        self.final_alpha = None  # 拉格朗日乘子结果
        self.K = None  # 特征经过核函数转化的值

    def fit(self, X_train, Y_train):
        self.kernel_process(X_train, Y_train, self.kernel)

        for now_iter in range(self.epoch):
            alpha_prev = np.copy(self.alpha)
            for j in range(self.m):

                # 选择第二个优化的拉格朗日乘子
                i = self.random_index(j)
                error_i, error_j = self.error_row(i, alpha_prev), self.error_row(j, alpha_prev)

                # 检验他们是否满足KKT条件，然后选择违反KKT条件最严重的self.alpha[j]
                if (self.Y[j] * error_j < -0.001 and self.alpha[j] < self.C) or (self.Y[j] * error_j > 0.001 and self.alpha[j] > 0):

                    eta = 2.0 * self.K[i, j] - self.K[i, i] - \
                        self.K[j, j]  # 第j个要优化的拉格朗日乘子，最后需要的

                    if eta >= 0:
                        continue

                    L, H = self.getBounds(i, j)
                    # 旧的拉格朗日乘子的值
                    old_alpha_j, old_alpha_i = self.alpha[j], self.alpha[i]
                    # self.alpha[j]的更新
                    self.alpha[j] -= (self.Y[j] * (error_i - error_j)) / eta

                    # 根据约束最后更新拉格朗日乘子self.alpha[j]，并且更新self.alpha[j]
                    self.alpha[j] = self.finalValue(self.alpha[j], H, L)
                    self.alpha[i] = self.alpha[i] + self.Y[i] * \
                        self.Y[j] * (old_alpha_j - self.alpha[j])

                    # 更新偏置值b
                    b1 = self.b - error_i - self.Y[i] * (self.alpha[i] - old_alpha_j) * self.K[i, i] - \
                        self.Y[j] * (self.alpha[j] -
                                     old_alpha_j) * self.K[i, j]
                    b2 = self.b - error_j - self.Y[j] * (self.alpha[j] - old_alpha_j) * self.K[j, j] - \
                        self.Y[i] * (self.alpha[i] -
                                     old_alpha_i) * self.K[i, j]
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)

            # 判断是否收敛(终止)
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.epsilon:
                self.final_alpha = np.copy(self.alpha)
                break

        self.final_alpha = np.copy(self.alpha)

    def predict(self, X):
        n = X.shape[0]
        result = np.zeros(n)
        for i in range(n):
            result[i] = np.sign(self.predict_row(X[i, :], self.final_alpha))  # 正的返回1，负的返回-1
        return result

    # 用带拉格朗日乘子表示的w代入wx+b
    def predict_row(self, X, alpha):
        k_v = self.kernel(self.X, X)
        return np.dot((alpha * self.Y).T, k_v.T) + self.b

    def kernel_process(self, X_train, Y_train, kernel):
        self.X = X_train
        self.Y = Y_train
        self.m = X_train.shape[0]
        self.n = X_train.shape[1]
        self.K = np.zeros((self.m, self.m))  # 核的新特征数组初始化
        self.alpha = np.zeros(self.m)  # 拉格朗日乘子初始化
        # self.bar = progressbar.ProgressBar(widgets=bar_widgets)  # 进度条
        if kernel == None:
            self.kernel = LinearKernel()  # 无核默认是线性的核
        else:
            self.kernel = kernel
        for i in range(self.m):
            self.K[:, i] = self.kernel(
                self.X, self.X[i, :])  # 每一行数据的特征通过核函数转化 n->m

    # 随机一个要优化的拉格朗日乘子，该乘子必须和循环里面选择的乘子不同
    def random_index(self, first_alpha):
        i = first_alpha
        while i == first_alpha:
            i = np.random.randint(0, self.m - 1)
        return i

    # 预测的值减真实的Y
    def error_row(self, i, alpha):
        return self.predict_row(self.X[i], alpha) - self.Y[i]

    # 得到self.alpha[j]的范围约束
    def getBounds(self, i, j):
        if self.Y[i] != self.Y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H

    # 根据self.alpha[i]的范围约束获得最终的值
    def finalValue(self, alpha, H, L):
        if alpha > H:
            alpha = H
        elif alpha < L:
            alpha = L
        return alpha


if __name__ == '__main__':
    use_sklearn = False

    # Data Preprocessing
    dataset = pd.read_csv('./dataset/Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    Y = dataset.iloc[:, 4].values

    # Making Dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train.astype(np.float64))
    X_test = sc.transform(X_test.astype(np.float64)) 

    if use_sklearn:
        from sklearn.svm import SVC
        classifier = SVC(kernel='linear', random_state=0)
        classifier.fit(X_train, Y_train)
    else:
        classifier = Skylark_SVC()
        classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    # Confusion matrix
    print_confusion_matrix(Y_test, Y_pred, clf_name='SVM')

    # Visualization
    visual_train(X_train, Y_train, classifier, clf_name='SVM')
