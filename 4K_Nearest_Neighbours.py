import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Skylark_KNeighborsClassifier():
    def __init__(self, n_neighbors, metric, p):
        super().__init__()
        self.n_neighbors = n_neighbors

    def fit(self, X, Y):
        ...

    def predict(self, X):
        ...


if __name__ == '__main__':
    use_sklearn = True

    # Data Preprocessing
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    Y = dataset.iloc[:, 4].values

    # Making Dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if use_sklearn:
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(
            n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(X_train, Y_train)
    else:
        classifier = Skylark_KNeighborsClassifier(
            n_neighbors=5, metric='minkowski', p=2
        )
        classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)
