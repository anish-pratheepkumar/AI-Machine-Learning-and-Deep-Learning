import numpy as np
from numpy import genfromtxt


def load_data(path, num_train):
    #####################################################
    # Load the data and return the matrices
    # X_train, Y_train, X_test, Y_test.
    # The matrices X_train and X_test have to contain
    # the data vectors as rowsincluding a '1' as the
    # last entry of each row.
    # The vectors Y_train and Y_test have to contain
    # the target values for each data vector.
    # Split the data into train and test sets after
    # num_train elements.
    ######################################################
    winedata = genfromtxt(path, delimiter=';', skip_header=1) #The index(1st) row is removed from data
    
    X = np.hstack((winedata[:, :-1], np.ones((winedata.shape[0], 1))))    # adding a column of 1(bias)
    Y = winedata[:, -1]                                              # selecting quality column as output
    
    X_train = X[:num_train]                                       #select training features
    Y_train = Y[:num_train]                                       #select corresponding training targets

    X_test = X[num_train:]
    Y_test = Y[num_train:]
    
    return X_train,Y_train,X_test,Y_test


def fit(X, Y):
    ######################################################
    # Return a vector theta containing the weights by
    # applying linear regression for data X and targets Y.
    ######################################################
    #Xtrans = X.T
    #Ytrans = Y.T
    temp = np.dot(Y.T,X)
    theta = np.dot(np.linalg.inv(np.dot(X.T,X)),temp.T)
    return theta


def predict(X, theta):
    ######################################################
    # Perform inference using data X and weights theta,
    # i.e. return the predicted values Y_pred.
    ######################################################
    Y_pred = np.dot(X, theta)
    return Y_pred
    pass


def energy(Y_pred, Y_gt):
    ######################################################
    # Compute the energy by calculating the sum over the
    # squared distances of Y_pred and Y_gt.
    ######################################################
    en = np.sum((np.square(Y_pred - Y_gt)))
    return en
