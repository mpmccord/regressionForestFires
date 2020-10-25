from regression import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import os
# Creates a labeled heatmap indexed by columns
def heatmap(arr, labels):
    fig, ax = plt.subplots()
    cov_matrix = np.cov(arr, rowvar=False)
    cov_matrix = cov_matrix / np.linalg.norm(cov_matrix)
    im = ax.imshow(cov_matrix)
    cbar = ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    ax.set_title("Covariance Matrix")
    fig.tight_layout()
    plt.show()
    # Graphs iris using my methods
def graphIris():
    iris = arr(getPath("iris.data"))
    names = getLabels(getPath("iris.data"))
    Y_pred = plotSimple(iris, names, 0, 1)
    
    Y_pred = plotQuadratic(iris, names, 0, 1)
    
# Plots the array wuth quadratic linear regression
def plotQuadratic(arr, names, Xindex, Yindex):
    print("Quadratic Linear Regression: ")
    heatmap(arr, names)
    area = names[Yindex]
    Y_quad = simpleQuadraticRegression(arr, Xindex, Yindex, labels=names)
    plotResiduals(arr, Y_quad)
    '''
    arr = normalize(arr)
    Y_quad = simpleQuadraticRegression(arr, 8, 12, names)
    plotResiduals(arr, Y_quad)
    '''
# Plots simple linear regression
def plotSimple(arr, labels, Xindex, Yindex):
    print("Simple Linear Regression")
    names = getLabels(getPath("forestfires.csv"))
    print(names)
    heatmap(arr, names)
    area = names[Yindex]
    Y_simple = simpleSingleRegression(arr, Xindex, Yindex, labels=names)
    plotResiduals(arr, Y_simple)
# Plots multiple linear regression (does not compile)
"""
def plotMultiple(arr, X1, X2, Y):
    X1_true = arr[:,X1].reshape(arr.shape[0], 1)
    X2_true = arr[:, X2].reshape(arr.shape[0], 1)
    Y_true = arr[:,Y].reshape(arr.shape[0], 1)

    # Tack a homogeneous coordinate (H) onto the independent variable (X)
    H = np.ones((X1.shape))
    A = np.hstack((X1,H))
    # Tack a homogeneous coordinate (H) onto the independent variable (X)
    H = np.ones((X_true.shape))
    A = np.hstack((X_true,H))
    # Linear Regression: Y = m*x + b
    W = np.linalg.inv( A.T @ A ) @ A.T @ Y_true
    Y_pred = A @ W
    print("Weights: ", W)
    """
"""
def graphForestFiresMultiple():
    fires = arr(getPath("forestfires.csv"))
    names = getLabels(getPath("forestfires.csv"))
    print(names)
    area = names[12]
    print(multiple_linear(fires, 8, 10, 12).summary())
    """
def graphClimateChange():
    fires = pd.read_csv("GlobalLandTemperaturesbyState.csv")
    fires = fires.groupby(pd.Grouper(key='dt', freq='A')).mean().dropna()
    fires = fires.to_numpy()
    names = getLabels(getPath("GlobalLandTemperaturesbyState.csv"))
    print(names)
    area = names[12]
    plotSimple(fires, names)
    plotQuadratic(fires, names)
def graphForestFires():
    fires = arr(getPath("forestfires.csv"))
    names = getLabels(getPath("forestfires.csv"))
    print(names)
    area = names[12]
    # plotSimple(fires, names)
    plotQuadratic(fires, names)
def main():
    graphClimateChange()
if __name__ == '__main__':
    print(main())