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
    print(mu)
    sigma = np.std(arr, axis=0).reshape(1, arr.shape[1])
    print(sigma)
    return arr @ scaleTranslate(arr, mu, sigma)
def scaleTranslate(arr, transMatrix, scaleMatrix):
    size = arr.shape[1]
    T = np.eye(size)
    print("Identity Matrix")
    print(T)
    T[0:-1, -1] = -transMatrix[0,0:-1]
    print("Trans matrix")
    print(transMatrix)
    S = np.eye(size)
    S[0:-1,0:-1] = S[0:-1,0:-1] * (1/scaleMatrix[0,0:-1])
    print("Scaled")
    print(S)
    N = S @ T
    return N
def plot_cov(arr):
    cov_matrix = np.cov(arr, rowvar=False)
    sns.heatmap(cov_matrix, annot=True)
    plt.show()
def simpleSingleRegression(arr, X, Y):
    X_true = arr[X][:]
    Y_true = arr[Y][:]
    n = X_true.shape[0]
    H_true = np.ones( (n,1) )
    A_simple = np.hstack( (X_true, H_true) )
    print("A:")
    print(A_simple[:3,:])

    # 2. Solve for the weights W_simple that optimize the model for polynomial degree D=1
    # 𝑊 = (𝐴.𝑇 𝐴)−1 𝐴.𝑇 𝑌
    W_simple = np.linalg.inv(A_simple.T @ A_simple) @ A_simple.T @ Y_true
    print("W:")
    print(W_simple)

    # 3. Make predictions
    Y_simple = A_simple @ W_simple

    # 4. Evaluate the fit's mean residual, RSS, and R squared
    R_simple = Y_true - Y_simple
    mean_r_simple = np.mean( R_simple )
    print("Mean residual:", mean_r_simple)

    RSS_simple = np.sum(R_simple**2)
    print("RSS:", RSS_simple)

    mean_true = np.mean(Y_true)
    SS_true = np.sum((Y_true - mean_true)**2)
    RSq_simple = 1 - RSS_simple / SS_true
    print("RSq:", RSq_simple)
def graphIris():
    iris = arr(getPath("iris.data"))
    names = getLabels(getPath("iris.data"))
    iris = normalize(iris)
    print(iris)
    # plot_cov(arr)
def main():
    test = np.arange(9).reshape(3, 3)
    print(test)
    test = normalize(test)
    print("Normalized")
    print(test)
if __name__ == '__main__':
    print(main())