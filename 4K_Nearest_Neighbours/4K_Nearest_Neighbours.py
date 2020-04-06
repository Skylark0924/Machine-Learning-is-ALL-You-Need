import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Skylark_KNeighborsClassifier():
    def __init__(self, k_neighbors):
        super().__init__()
        self.k_neighbors = k_neighbors

    def predict(self, X_test, X_train, y_train):
        y_predict = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            distances = np.zeros((X_train.shape[0], 2))  # 测试的数据和训练的各个数据的欧式距离
            for j in range(X_train.shape[0]):
                dis = self.euclidean_distance(X_test[i], X_train[j])  # 计算欧式距离
                label = y_train[j]  # 测试集到的每个训练集的数据的分类标签
                distances[j] = [dis, label]

                # argsort()得到测试集到训练的各个数据的欧式距离从小到大排列并且得到序列，然后再取前k个.
                k_nearest_neighbors = distances[distances[:, 0].argsort(
                )][:self.k_neighbors]

                # 利用np.bincount统计k个近邻里面各类别出现的次数
                counts = np.bincount(k_nearest_neighbors[:, 1].astype('int'))

                # 得出每个测试数据k个近邻里面各类别出现的次数最多的类别
                testLabel = counts.argmax()
                y_predict[i] = testLabel

        return y_predict

    def euclidean_distance(self, x1, x2):
        """ Calculates the l2 distance between two vectors """
        distance = 0
        # Squared distance between each coordinate
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
        return math.sqrt(distance)


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
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(
            n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
    else:
        classifier = Skylark_KNeighborsClassifier(
            k_neighbors=5
        )
        Y_pred = classifier.predict(X_test, X_train, Y_train)

    # Y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)
