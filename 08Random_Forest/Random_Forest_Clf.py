import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
sys.path.append("D:\Github\Machine-Learning-Basic-Codes")
sys.path.append("D:/Github/Machine-Learning-Basic-Codes/07Decision_Trees")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from Decision_Trees_Clf import Skylark_DecisionTreeClassifier
from utils.visualize import *

class Skylark_RandomForestClassifier():
    def __init__(self, n_estimators=100, min_samples_split=2, min_gain=0,
                 max_depth=float("inf"), max_features=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.max_features = max_features

        self.trees = []
        # 建立森林(bulid forest)
        for _ in range(self.n_estimators):
            tree = Skylark_DecisionTreeClassifier(min_samples_split=self.min_samples_split, min_impurity=self.min_gain,
                                                  max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, X_train, Y_train):
        # 训练，每棵树使用随机的数据集(bootstrap)和随机的特征
        # every tree use random data set(bootstrap) and random feature
        sub_sets = self.get_bootstrap_data(X_train, Y_train)
        n_features = X_train.shape[1]
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            # 生成随机的特征
            # get random feature
            sub_X, sub_Y = sub_sets[i]
            idx = np.random.choice(n_features, self.max_features, replace=True)
            sub_X = sub_X[:, idx]
            self.trees[i].fit(sub_X, sub_Y)
            self.trees[i].feature_indices = idx
            print("tree", i, "fit complete")

    def predict(self, X_test):
        y_preds = []
        for i in range(self.n_estimators):
            idx = self.trees[i].feature_indices
            sub_X = X_test[:, idx]
            y_pre = self.trees[i].predict(sub_X)
            y_preds.append(y_pre)
        y_preds = np.array(y_preds).T
        y_pred = []
        for y_p in y_preds:
            # np.bincount()可以统计每个索引出现的次数
            # np.argmax()可以返回数组中最大值的索引
            # cheak np.bincount() and np.argmax() in numpy Docs
            y_pred.append(np.bincount(y_p.astype('int')).argmax())
        return np.array(y_pred)

    def get_bootstrap_data(self, X, Y):
        # 通过bootstrap的方式获得n_estimators组有放回的采样数据
        # get int(n_estimators) datas by bootstrap
        m = X.shape[0]
        Y = Y.reshape(m, 1)

        # 合并X和Y，方便bootstrap (conbine X and Y)
        X_Y = np.hstack((X, Y))
        np.random.shuffle(X_Y)

        data_sets = []
        for _ in range(self.n_estimators):
            idm = np.random.choice(m, m, replace=True)
            bootstrap_X_Y = X_Y[idm, :]
            bootstrap_X = bootstrap_X_Y[:, :-1]
            bootstrap_Y = bootstrap_X_Y[:, -1:]
            data_sets.append([bootstrap_X, bootstrap_Y])
        return data_sets


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
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(
            n_estimators=10, criterion='entropy', random_state=0)
        classifier.fit(X_train, Y_train)
    else:
        classifier = Skylark_RandomForestClassifier(n_estimators=10)
        classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    print_confusion_matrix(
        Y_test, Y_pred, clf_name='Random Forest Classification')

    # Visualising the Training set results
    visualization_clf(X_train, Y_train, classifier,
                      clf_name='Random Forest Classification', set_name='Training')
    # Visualising the Test set results
    visualization_clf(X_test, Y_test, classifier,
                      clf_name='Random Forest Classification', set_name='Test')