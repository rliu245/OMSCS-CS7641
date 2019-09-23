# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 03:18:19 2019

@author: Ray
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn import datasets

def PlotData(x, y, x2, y2):
    plt.figure()
    plt.plot(x, y, 'r', x2, y2, 'g')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    
    red_patch = mpatches.Patch(color='red', label='train')
    green_path = mpatches.Patch(color='green', label = 'test')
    plt.legend(handles=[red_patch, green_path])

    plt.show()
    
if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features. We could
                          # avoid this ugly slicing by using a two-dim dataset
    y = iris.target
    PlotData(X, y, X, y)