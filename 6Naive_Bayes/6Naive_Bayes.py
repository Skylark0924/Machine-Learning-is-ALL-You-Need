import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("D:\Github\Machine-Learning-Basic-Codes")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.visualize import *

class Skylark_Naive_Bayes():
    def __init__(self):
        super().__init__()

    def fit(self, X_train, Y_train):
        self.X = X_train
        self.y = Y_train
        self.classes = np.unique(Y_train)
        self.parameters = {}
        for i, c in enumerate(self.classes):
            # 计算每个种类的平均值，方差，先验概率
            X_Index_c = X[np.where(Y_train == c)]
            X_index_c_mean = np.mean(X_Index_c, axis=0, keepdims=True)
            X_index_c_var = np.var(X_Index_c, axis=0, keepdims=True)
            parameters = {"mean": X_index_c_mean, "var": X_index_c_var, "prior": X_Index_c.shape[0] / X.shape[0]}
            self.parameters["class" + str(c)] = parameters

    def predict(self, X_test):
        # 取概率最大的类别返回预测值
        output = self._predict(X_test)
        output = np.reshape(output, (self.classes.shape[0], X_test.shape[0]))
        prediction = np.argmax(output, axis=0)
        return prediction

    def _predict(self, X_test):
        # 计算每个种类的概率P(Y|x1,x2,x3) =  P(Y)*P(x1|Y)*P(x2|Y)*P(x3|Y)
        output = []
        for y in range(self.classes.shape[0]):
            prior = np.log(self.parameters["class" + str(y)]["prior"])
            posterior = self._pdf(X_test, y)
            prediction = prior + posterior
            output.append(prediction)
        return output

    def _pdf(self, X, classes):
        # 一维高斯分布的概率密度函数
        # eps为防止分母为0
        eps = 1e-4
        mean = self.parameters["class" + str(classes)]["mean"]
        var = self.parameters["class" + str(classes)]["var"]

        # 取对数防止数值溢出
        # numerator.shape = [m_sample,feature]
        numerator = np.exp(-(X - mean) ** 2 / (2 * var + eps))
        denominator = np.sqrt(2 * np.pi * var + eps)

        # 朴素贝叶斯假设(每个特征之间相互独立)
        # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y),取对数相乘变为相加
        # result.shape = [m_sample,1]
        result = np.sum(np.log(numerator / denominator), axis=1, keepdims=True)
        return result.T


if __name__ == '__main__':
    use_sklearn = False

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

    if use_sklearn:
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB(priors=None, var_smoothing=1e-09)
        classifier.fit(X_train, Y_train)
    else:
        classifier = Skylark_Naive_Bayes()
        classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    print_confusion_matrix(
        Y_test, Y_pred, clf_name='Naive Bayes Classification')

    # Visualising the Training set results
    visualization(X_train, Y_train, classifier,
                  clf_name='Naive Bayes Classification', set_name='Training')

    # Visualising the Test set results
    visualization(X_train, Y_train, classifier,
                  clf_name='Naive Bayes Classification', set_name='Test')
