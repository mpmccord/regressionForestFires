import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import os
import math
# Gets the file path given a file name in the format myFile.filetype
# Note: this assumes that the file is in the same directory as main.py
def getPath(myfile):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, myfile)
    return file_path

# Creates a numpy array given a filename
# This assumes that there is one header line to skip and that
# the file is a csv or otherwise separated by commas
def arr(myFile):
    arr = np.genfromtxt(getPath(myFile), skip_header=1, delimiter=',')
    ones = np.ones( (arr.shape[0],arr.shape[1]) )
    A = np.hstack( (arr, ones) )
    return arr
# Note: this requires X to be a specific label, plots pair plots against that label
def pairplot(arr, labels, X):
    df = pd.DataFrame(arr, columns=labels)
    sns.pairplot(df, x_vars=labels, y_vars=X, kind='reg')
# plots a scatterplot of the array and its labels
def scatterplot(arr, c1, c2, labels):
    plt.xlabel(labels[c1])
    plt.ylabel(labels[c2])
    sns.scatterplot(arr[c1], arr[c2])
    plt.show()
# scales and translates a matrix to normalize it
def scaleTranslate(arr, transMatrix, scaleMatrix):
    size = arr.shape[1]
    T = np.eye(size)
    T[0:-1, -1] = -transMatrix[0,0:-1]

    S = np.eye(size)
    S[0:-1,0:-1] = S[0:-1,0:-1] * (1/scaleMatrix[0,0:-1])
    N = S @ T
    return N
# Shows the stats of the array
def showStats(arr):
    print("Standard Deviation: ")
    std = np.std(arr, axis=0)
    print(std)
    plt.plot(std, '--or')
    print("Covariance: ")
    cov = np.cov(arr, rowvar=False)
    plt.plot(cov, '--ob')
    print("Mean")
    means = np.mean(arr, axis=0)
    plt.plot(means, '--om')

# Gets the first row of the file and returns a list of the names of the columns
# Assumes that the first row is separated by commas
def getLabels(filename):
    myFile = open(getPath(filename), "r")
    labels = myFile.readline()
    labels = labels.split(",")
    myFile.close()
    return labels
# I initially implemented this using from scratch methods but ran into bugs and now
# use built in methods
def normalize(arr):
    """
    mu = np.mean(arr, axis=0).reshape(1, arr.shape[1])
    sigma = np.std(arr, axis=0).reshape(1, arr.shape[1])
    """
    return arr / np.linalg.norm(arr)

def simpleQuadraticRegression(arr, X, Y, labels):
    X_true = arr[:,X].reshape(arr.shape[0], 1)
    Y_true = arr[:,Y].reshape(arr.shape[0], 1)

    # Tack a homogeneous coordinate (H) onto the independent variable (X)
    H = np.ones((X_true.shape))
    A = np.hstack((X_true,H))
    # Tack a homogeneous coordinate (H) onto the independent variable (X)
    H = np.ones((X_true.shape))
    A = np.hstack((X_true ** 2, X_true, H))
    # Linear Regression: Y = m*x + b
    W = np.linalg.inv( A.T @ A ) @ A.T @ Y_true
    Y_pred = A @ W
    print("Weights", W)
    # Visualization
    plotLine(X_true, Y_pred, Y, X, Y_true, labels)
    return Y_pred
def simpleSingleRegression(arr, X, Y, labels):
    X_true = arr[:,X].reshape(arr.shape[0], 1)
    Y_true = arr[:,Y].reshape(arr.shape[0], 1)

    # Tack a homogeneous coordinate (H) onto the independent variable (X)
    H = np.ones((X_true.shape))
    A = np.hstack((X_true,H))
    # Tack a homogeneous coordinate (H) onto the independent variable (X)
    H = np.ones((X_true.shape))
    A = np.hstack((X_true,H))
    # Linear Regression: Y = m*x + b
    W = np.linalg.inv( A.T @ A ) @ A.T @ Y_true
    Y_pred = A @ W
    print("Weights: ", W)
    # Visualization
    plotLine(X_true, Y_pred, Y, X, Y_true, labels)
    return Y_pred
# Plots the residuals and prints their stats
def residuals(Y_true, Y_pred):
    R_simple = Y_true - Y_pred
    mean_r_simple = np.mean( R_simple )
    print("Mean residual:", mean_r_simple)

    RSS_simple = np.sum(R_simple**2)
    print("RSS:", RSS_simple)

    mean_true = np.mean(Y_true)
    SS_true = np.sum((Y_true - mean_true)**2)
    RSq_simple = 1 - RSS_simple / SS_true
    print("RSq:", RSq_simple)
    return np.abs(R_simple)
# Plots the data points and a line given an X matrix, a Y matrix and prediction
def plotLine(X_true, Y_pred, Y, X, Y_true, labels):
    # Visualization
    plt.plot(X_true,Y_true, 'ob', label="known")
    plt.plot( X_true, Y_pred, 'or', linewidth=5, label="predicted" )
    plt.xlabel(labels[X])
    plt.ylabel(labels[Y])
    plt.grid()
    plt.legend()
    plt.show()
# Plots residuals given Y values and expected Y values
def plotResiduals(Y_true, Y_pred):
    plt.title("Residuals")
    resid = residuals(Y_true[:,0], Y_pred[:, 0])
    plt.plot(resid, 'ob')
    plt.show()
# Testing my methods against a simple plot
def test():
    myTest = np.arange(81).reshape(27, 3)
    myTest[:,1] = myTest[:, 0] ** 10 + 12 + (np.sin(myTest[:, 0]) * 30)
    plt.scatter(myTest[:,0], myTest[:,1])
    plt.show()
    test = myTest
    labels = ["myx", "myy", "myz"]
    # test = normalize(myTest)
    
    Y_pred = simpleQuadraticRegression(test, 0, 1, labels)
    resid = residuals(test[:, 0], Y_pred)
    plotResiduals(test, Y_pred)
if __name__ == '__main__':
      (test())