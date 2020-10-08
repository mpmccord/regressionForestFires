from regression import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import os

# Applying my methods to iris
def graphIris():
    iris = arr(getPath("iris.data"))
    names = getLabels(getPath("iris.data"))
    iris = normalize(iris)
    print(iris)

def main():
    print("Hello world")
if __name__ == '__main__':
    print(main())