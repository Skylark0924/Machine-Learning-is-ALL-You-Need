import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
import sys
sys.path.append("D:/Github/Machine-Learning-Basic-Codes")

from sklearn.preprocessing import StandardScaler

from utils.visualize import *

class Skylark_PCA():
    def __init__(self, n_components):
        self.K = n_components

    def fit(self, X_train):
        ...

    def transform(self, X_train):
        U = self.SVD_decompose(X_train)
        Z = np.zeros((X_train.shape[0],self.K))
        Ureduce = U[:, 0:self.K]
        Z = np.dot(X_train, Ureduce)
        self.components_ = np.transpose(Ureduce)
        return Z

    def SVD_decompose(self, X_train):
        m=X_train.shape[0]
        Sigma = np.dot(np.transpose(X_train), X_train)/m
        U,S,V = np.linalg.svd(Sigma)
        return U


def PCA_load_data(data_name):
    if data_name == 'data':
        '''加载数据并作图'''
        data = spio.loadmat('./dataset/PCA/data.mat')
        X = data['X']
        plt = plot_data_2d(X,'bo')
        plt.axis('square')
        plt.title('original data')
        plt.show()
    elif data_name == 'data_faces':
        '''加载数据并显示'''
        image_data = spio.loadmat('./dataset/PCA/data_faces.mat')
        X = image_data['X']
        display_imageData(X[0:100,:])  # 显示100个最初图像
    else:
        print('Undefined Dataset!')

    '''归一化数据并作图'''
    scaler = StandardScaler()
    scaler.fit(X)
    x_train = scaler.transform(X)
    
    if data_name == 'data':
        plot_data_2d(x_train, 'bo')
        plt.axis('square')
        plt.title('scaler data')
        plt.show()
    return x_train

def PCA_result(Z, data_name):
    '''数据恢复并作图'''
    Ureduce = model.components_     # 得到降维用的Ureduce
    x_rec = np.dot(Z, Ureduce)       # 数据恢复
    
    if data_name == 'data':
        plot_data_2d(x_rec,'bo')
        plt.plot()
        plt.axis('square')
        plt.title('recover data')
        plt.show()
    elif data_name == 'data_face':
        display_imageData(Ureduce[0:36,:])  # 可视化部分U数据
        display_imageData(x_rec[0:100,:])  # 显示恢复的数据


if __name__ == "__main__":
    use_sklearn = False
    data_name = 'data'

    K=1 # 降维后的维度
    X_train = PCA_load_data(data_name)

    if use_sklearn:
        from sklearn.decomposition import pca
        model = pca.PCA(n_components=K)
        model.fit(X_train)
    else:
        model = Skylark_PCA(n_components=K)
        model.fit(X_train)

    Z = model.transform(X_train)    # transform就会执行降维操作
    PCA_result(Z, data_name)

