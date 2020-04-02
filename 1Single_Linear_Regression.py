import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Preprocessing
dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

if use_sklearn:
    from sklearn.cross_validation import train_test_split
    from sklearn.linear_model import LinearRegression

    # Making Dataset
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 

    # Fitting Simple Linear Regression Model to the training set
    regressor = LinearRegression()
    regressor = regressor.fit(X_train, Y_train)

    # Predecting the Result
    Y_pred = regressor.predict(X_test)

    # Visualization
    ## Training Results
    plt.scatter(X_train , Y_train, color = 'red')
    plt.plot(X_train , regressor.predict(X_train), color ='blue')
    ## Testing Results
    plt.scatter(X_test , Y_test, color = 'red')
    plt.plot(X_test , regressor.predict(X_test), color ='blue')

else:
    class Skylark_LinearRegression():
        def __init__():
            ...
        def 
        

if __name__=='__main__':
    use_sklearn==True
