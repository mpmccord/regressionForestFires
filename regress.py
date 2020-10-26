import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import os
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

# Gets the first row of the file and returns a list of the names of the columns
# Assumes that the first row is separated by commas
def getLabels(filename):
    myFile = open(getPath(filename), "r")
    labels = myFile.readline()
    labels = labels.split(",")
    myFile.close()
    return labels
def normalize(arr):
    mu = np.mean(arr, axis=0).reshape(1, arr.shape[1])
    sigma = np.std(arr, axis=0).reshape(1, arr.shape[1])
    return arr @ scaleTranslate(arr, mu, sigma)
def scaleTranslate(arr, transMatrix, scaleMatrix):
    size = arr.shape[1]
    T = np.eye(size)
    T[0:-1, -1] = -transMatrix[0,0:-1]
    S = np.eye(size)
    S[0:-1,0:-1] = S[0:-1,0:-1] * (1/scaleMatrix[0,0:-1])
    ("Scaled")
    N = S @ T
    return N
def plot_cov(arr):
    cov_matrix = np.cov(arr, rowvar=False)
    sns.heatmap(cov_matrix, annot=True)
    plt.show()
# Converts x and y into matrices and
def preprocess(arr, X, Y):
    X = arr[:,X].reshape(arr.shape[0], 1)
    Y = arr[:,Y].reshape(arr.shape[0], 1)

    # Tack a homogeneous coordinate (H) onto the independent variable (X)
    H = np.ones((X.shape))
    A = np.hstack((X,H))
    return X, Y, A, H
def linearWeights(A, X, Y):
    # Linear Regression: Y = m*x + b
    W = np.linalg.lstsq(A, Y, rcond=None)[0]
    Y_pred = A @ W
    return Y_pred
def cubicWeights(X, Y, H):
    A_cubic = np.hstack((X ** 3, X ** 2, X, H))
    print(A_cubic.shape)
    print(Y.shape)
    # 2. Solve for the weights W_cubic that optimize the model for polynomial degree D=3
    # W_simple = np.linalg.inv(A_simple.T @ A_simple) @ A_simple.T @ Y_true
    print(A_cubic.T.shape)
    W_cubic = np.linalg.inv( A_cubic.T @ A_cubic ) @ A_cubic.T @ Y
    print("W", W_cubic.shape)
    print("Weights", W_cubic.shape, W_cubic)
    Y_pred = A_cubic @ W_cubic
    return Y_pred
def simpleCubicRegression(arr, X, Y):
    X, Y, A, H = preprocess(arr, X, Y)
    Y_pred = cubicWeights(A, X, Y)
    # Visualization
    plotLine(X, Y_pred, Y)
def simpleSingleRegression(arr, X, Y):
    X, Y, A, H = preprocess(arr, X, Y)
    Y_pred = linearWeights(A, X, Y)
    # Visualization
    plotLine(X, Y_pred, Y)


# Plots the data points and a line given an X matrix, a Y matrix and prediction
def plotLine(X, Y_pred, Y):
    # Visualization
    plt.plot( X[:,0], Y[:,0], "ok", label="Y" )
    plt.plot( X[:,0], Y_pred[:,0], '-r', linewidth=3, label="Y pred")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.show()
# Testing my methods against a simple plot
def test():
    myTest = np.arange(81).reshape(27, 3)
    # print(myTest)
    myTest[:,1] = myTest[:, 0] ** 3 + 12
    test = myTest
    labels = ["x", "y", "z"]
    # test = normalize(myTest)
    Y_pred = simpleSingleRegression(test, 0, 1)
    print(Y_pred)
    Y_pred = simpleCubicRegression(test, 0, 1)
    print(Y_pred)
if __name__ == '__main__':
      (test())