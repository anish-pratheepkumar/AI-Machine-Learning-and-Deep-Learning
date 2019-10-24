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
    X_temp1 = winedata[0:num_train,:11]                       #selecting 3674 rows and excluding                                                                        quality(output) column
    X_train = np.ones((X_temp1.shape[0], X_temp1.shape[1]+1)) #creating a matrix of ones
    X_train[:,:-1] = X_temp1                                  #adding a column of 1(bias)
    Y_train = winedata[0:num_train,11:]                       #selecting quality column as output data
    #following similar procedure to create test set of 1224 samples
    X_temp2 = winedata[num_train:,:11]
    X_test = np.ones((X_temp2.shape[0], X_temp2.shape[1]+1))
    X_test[:,:-1] = X_temp2
    Y_test = winedata[num_train:,11:]
    return (X_train,Y_train,X_test,Y_test)


def fit(X, Y):
    ######################################################
    # Return a vector theta containing the weights by
    # applying linear regression for data X and targets Y.
    ######################################################
    Xtrans = X.transpose()
    Ytrans = Y.transpose()
    temp = np.dot(Ytrans,X)
    theta = np.dot(np.linalg.inv(np.dot(Xtrans,X)),temp.transpose())
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
