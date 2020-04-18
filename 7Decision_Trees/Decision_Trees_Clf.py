import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("D:\Github\Machine-Learning-Basic-Codes")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.visualize import *
from utils.tool_func import *
from super_class import *

class Skylark_DecisionTreeClassifier(DecisionTree):
    '''
    分类树
    '''
    def _calculate_information_gain(self, y, y1, y2):
        # Calculate information gain
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * \
                                                      calculate_entropy(y2)
        # print("info_gain",info_gain)
        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        # print("most_common :",most_common)
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(Skylark_DecisionTreeClassifier, self).fit(X, y)

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
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(
            criterion='entropy', random_state=0)
        classifier.fit(X_train, Y_train)
    else:
        classifier = Skylark_DecisionTreeClassifier()
        classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    print_confusion_matrix(
        Y_test, Y_pred, clf_name='Decision Tree Classification')

    # Visualising the Training set results
    visualization_clf(X_train, Y_train, classifier,
                  clf_name='Decision Tree Classification', set_name='Training')

    # Visualising the Test set results
    visualization_clf(X_test, Y_test, classifier,
                  clf_name='Decision Tree Classification', set_name='Test')
