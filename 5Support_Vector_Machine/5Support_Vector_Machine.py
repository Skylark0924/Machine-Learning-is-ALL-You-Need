import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Skylark_SVC():
    def __init__(self, kernel, random_state):
        super().__init__()
        self.kernel = kernel

    def fit(self, X, Y):
        ...

    def predict(self, X):
        ...
    
    def kernelTrans(self, ):
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

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    if use_sklearn:
        from sklearn.svm import SVC
        classifier = SVC(kernel='linear', random_state=0)
        classifier.fit(X_train, Y_train)
    else:
        classifier = Skylark_SVC(kernel='linear', random_state=0)
        classifier.fit(X_train, Y_train)

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
