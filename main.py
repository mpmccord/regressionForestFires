from regression import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import os
def plotResiduals(arr, Y_pred):
    plt.title("Residuals")
    resid = residuals(arr[:,0], Y_pred[:, 0])
    plt.plot(resid, 'ob')
    plt.show()
# Applying my methods to iris
def graphIris():
    iris = arr(getPath("iris.data"))
    names = getLabels("iris.data")
    names = getLabels(getPath("iris.data"))
    Y_simple = simpleSingleRegression(iris, 0, 2)
    Y_quad = simpleQuadraticRegression(iris, 0, 2)
    plotResiduals(iris, Y_simple)
    plotResiduals(iris, Y_quad)
    iris = normalize(iris)
    simpleQuadraticRegression(iris, 0, 2)
    simpleSingleRegression(iris, 0, 2)

def main():
    graphIris()
if __name__ == '__main__':
    print(main())