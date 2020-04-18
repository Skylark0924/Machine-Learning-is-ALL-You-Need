from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def visualization_clf(X_train, Y_train, classifier, clf_name, set_name):
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
    plt.title('{} ({} set)'.format(clf_name, set_name))
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

def visualization_reg(X, Y, regressor, reg_name, set_name):
    plt.scatter(X, Y, color='red')
    plt.plot(X, regressor.predict(X), color='blue')
    plt.title('{} ({} set)'.format(reg_name, set_name))
    plt.show()

def print_confusion_matrix(Y_test, Y_pred, clf_name):
    cm = confusion_matrix(Y_test, Y_pred)
    print('{}:\n'.format(clf_name), cm)


def print_mse(Y_test, Y_pred, reg_name):
    mse = mean_squared_error(Y_test, Y_pred)
    print("Mean Squared Error for {}:".format(reg_name), mse)


def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse
