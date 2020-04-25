

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/home/skylark/Github/Machine-Learning-Basic-Codes")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.visualize import *



if __name__ == '__main__':
    use_xgboost_api = True

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

    if use_xgboost_api:
        import xgboost as xgb
        from xgboost import XGBClassifier
        classifier = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
        classifier.fit(X_train, Y_train)
    else:
        classifier = Skylark_Naive_Bayes()
        classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    print_confusion_matrix(
        Y_test, Y_pred, clf_name='Naive Bayes Classification')

    # Visualising the Training set results
    visualization_clf(X_train, Y_train, classifier,
                  clf_name='Naive Bayes Classification', set_name='Training')
    # Visualising the Test set results
    visualization_clf(X_test, Y_test, classifier,
                  clf_name='Naive Bayes Classification', set_name='Test')
