import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("D:\Github\Machine-Learning-Basic-Codes")
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utils.visualize import *
from utils.tool_func import *
from super_class import *

class Skylark_DecisionTreeRegressor(DecisionTree):
    '''
    回归树
    '''
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        # Calculate the variance reduction
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

        return sum(variance_reduction)

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(Skylark_DecisionTreeRegressor, self).fit(X, y)


if __name__ == '__main__':
    use_sklearn = True

# Data Preprocessing
    dataset = pd.read_csv('./dataset/studentscores.csv')
    X = dataset.iloc[:, : 1].values
    Y = dataset.iloc[:, 1].values

    # Making Dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1/4, random_state=0)

    if use_sklearn:
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(
            criterion='mse', random_state=0)
        regressor.fit(X_train, Y_train)
    else:
        regressor = Skylark_DecisionTreeRegressor()
        regressor.fit(X_train, Y_train)

    Y_pred = regressor.predict(X_test)
    
    # MSE
    print_mse(Y_test, Y_pred, reg_name='DTR')

    # Visualization
    visualization_reg(X_train, Y_train, regressor,
                  reg_name='Decision Tree Regression', set_name='Training')
    visualization_reg(X_test, Y_test, regressor,
                  reg_name='Decision Tree Regression', set_name='Test')




