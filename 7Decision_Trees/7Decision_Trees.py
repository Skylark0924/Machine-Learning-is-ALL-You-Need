import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("D:\Github\Machine-Learning-Basic-Codes")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.visualize import *

class ClassificationTree():
    def __init__(self):
        super().__init__()

class Skylark_DecisionTreeClassifier():
    def __init__(self):
        super().__init__()

    def fit(self, X_train, Y_train):
        
    def predict(self, X_test):
        ...

if __name__ == '__main__':
    use_sklearn = True

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
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(
            criterion='entropy', random_state=0)
        classifier.fit(X_train, Y_train)
    else:
        classifier = Skylark_DecisionTreeClassifier(        )
        classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    print_confusion_matrix(
        Y_test, Y_pred, clf_name='Decision Tree Classification')

    # Visualising the Training set results
    visualization(X_train, Y_train, classifier,
                  clf_name='Decision Tree Classification', set_name='Training')

    # Visualising the Test set results
    visualization(X_train, Y_train, classifier,
                  clf_name='Decision Tree Classification', set_name='Test')
