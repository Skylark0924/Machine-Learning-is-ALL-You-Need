import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Skylark_LogisticRegression():
    def __init__(self):
        super().__init__()
        self.init_theta = np.zeros(3)  # X(m*n) so theta is n*1
        self.final_theta = np.zeros(3)

    def fit(self, X, Y):
        import scipy.optimize as opt
        res = opt.minimize(fun=self.cost, x0=self.init_theta, args=(
            X, Y), method='Newton-CG', jac=self.gradient)
        self.final_theta = res.x

    def predict(self, X):
        prob = self.sigmoid(X @ self.final_theta)
        return (prob >= 0.5).astype(int)

    def cost(self, theta, X, y):
        ''' cost fn is -l(theta) for you to minimize'''
        return np.mean(-y * np.log(self.sigmoid(X @ theta)) - (1 - y) * np.log(1 - self.sigmoid(X @ theta)))

    def gradient(self, theta, X, y):
        '''just 1 batch gradient'''
        return (1 / len(X)) * X.T @ (self.sigmoid(X @ theta) - y)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    use_sklearn = True

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
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression()
        classifier.fit(X_train, Y_train)
    else:
        classifier = Skylark_LogisticRegression()
        classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)
