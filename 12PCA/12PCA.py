import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
import sys
sys.path.append("/home/skylark/Github/Machine-Learning-Basic-Codes")

from sklearn.preprocessing import StandardScaler

from utils.visualize import *

def PCA_2d_data():
    '''加载数据并作图'''
    data = spio.loadmat('./dataset/PCA/data.mat')
    X = data['X']
    plt = plot_data_2d(X,'bo')
    plt.axis('square')
    plt.title('original data')
    plt.show()
    '''归一化数据并作图'''
    scaler = StandardScaler()
    scaler.fit(X)
    x_train = scaler.transform(X)
    
    plot_data_2d(x_train, 'bo')
    plt.axis('square')
    plt.title('scaler data')
    plt.show()

    return x_train
    

if __name__ == "__main__":
    use_sklearn = True

    K=1 # 要降的维度
    X_train = PCA_2d_data()

    if use_sklearn:
        from sklearn.decomposition import pca
        model = pca.PCA(n_components=K).fit(X_train)

