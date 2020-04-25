from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
from sklearn.metrics import confusion_matrix

import progressbar

bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]


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


# 可视化二维数据
def plot_data_2d(X,marker):
    plt.plot(X[:,0],X[:,1],marker) 
    return plt

# 显示图片
def display_imageData(imgData):
    sum = 0
    '''
    显示100个数（若是一个一个绘制将会非常慢，可以将要画的图片整理好，放到一个矩阵中，显示这个矩阵即可）
    - 初始化一个二维数组
    - 将每行的数据调整成图像的矩阵，放进二维数组
    - 显示即可
    '''
    m,n = imgData.shape
    width = np.int32(np.round(np.sqrt(n)))
    height = np.int32(n/width);
    rows_count = np.int32(np.floor(np.sqrt(m)))
    cols_count = np.int32(np.ceil(m/rows_count))
    pad = 1
    display_array = -np.ones((pad+rows_count*(height+pad),pad+cols_count*(width+pad)))
    for i in range(rows_count):
        for j in range(cols_count):
            max_val = np.max(np.abs(imgData[sum,:]))
            display_array[pad+i*(height+pad):pad+i*(height+pad)+height,pad+j*(width+pad):pad+j*(width+pad)+width] = imgData[sum,:].reshape(height,width,order="F")/max_val    # order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
            sum += 1
            
    plt.imshow(display_array,cmap='gray')   #显示灰度图像
    plt.axis('off')
    plt.show()

def kmeans_plot(X, centroids):
    plt.scatter(X[:,0], X[:,1])     # 原数据的散点图
    plt.plot(centroids[:,0],centroids[:,1],'r^',markersize=10)  # 聚类中心
    plt.show()
