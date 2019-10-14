# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:22:45 2019

@author: Ray
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import preprocess, mlp_plot

data = pd.read_csv('adult.test')
print(len(data))

for column in data.columns:
    indexNames = data[data[column] == ' ?'].index
    # Delete these row indexes from dataFrame
    data.drop(indexNames, inplace=True)
print(len(data))

data, data_test = train_test_split(data, train_size = 0.8)

X_train, y_train = data.drop('income', axis = 1), data['income']
X_test, y_test = data_test.drop('income', axis = 1), data_test['income']

X_train, y_train = preprocess.PreProcessIncome(X_train, y_train.values.ravel())
X_test, y_test = preprocess.PreProcessIncome(X_test, y_test.values.ravel())

""" Convert training and test data from Pandas to Numpy """
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

# pred_score_rhc, pred_score_sa, pred_score_ga = mlp_plot.MultiLayerPerceptronPlot(X_train, y_train, X_test, y_test)
_, train_errors, test_errors = mlp_plot.MLP_rhc(X_train, y_train, X_test, y_test)
mlp_plot.MLP_sa(X_train, y_train, X_test, y_test)
mlp_plot.MLP_ga(X_train, y_train, X_test, y_test)