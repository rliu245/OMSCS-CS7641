# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:22:45 2019

@author: Ray
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from modules import preprocess, feature_importance, plot_grid_search, svm_plot, mlp_plot, plot_data

data = pd.read_csv('./data/adult.data')
data_test = pd.read_csv('./data/adult.test')
print(len(data))

for column in data.columns:
    indexNames = data[data[column] == ' ?'].index
    # Delete these row indexes from dataFrame
    data.drop(indexNames, inplace=True)
print(len(data))

for column in data_test.columns:
    indexNames = data_test[data_test[column] == ' ?'].index
    # Delete these row indexes from dataFrame
    data_test.drop(indexNames, inplace=True)

X_train, y_train = data.drop('income', axis = 1), data['income']
X_test, y_test = data_test.drop('income', axis = 1), data_test['income']

X_train, y_train = preprocess.PreProcessIncome(X_train, y_train.values.ravel())
X_test, y_test = preprocess.PreProcessIncome(X_test, y_test.values.ravel())

# Compute the mathematics to obtain the scoring for every feature to determine which is important
test = feature_importance.FeatureImportance(np.array(X_train), np.array(y_train))
importantFeatures = []
for idx, column in enumerate(X_train.columns):
    importantFeatures.append((column, test[idx]))

""" Convert training and test data from Pandas to Numpy """
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

clf = DecisionTreeClassifier(random_state=63)
title = 'Decision Tree Classifier using multiple scorers simultaneously'
params = {'max_depth': range(1, 52, 2)}
plot_grid_search.GridSearch(clf, X_train, y_train, title, params)

svm_plot.SvmPlot(X_train, y_train, X_test, y_test)
    
clf = KNeighborsClassifier(weights = 'distance')
title = 'K Nearest Neighbor Classifier using multiple scorers simultaneously'
params = {'n_neighbors': list(range(1, 50))}
plot_grid_search.GridSearch(clf, X_train, y_train, title, params)

mlp_plot.MultiLayerPerceptronPlot(X_train, y_train, X_test, y_test)

max_depth_storage_train = []
max_depth_storage_test = []
acc_storage_train = []
acc_storage_test = []

for i in range(1, 52, 2):
    clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = i), random_state=63)
    clf.fit(X_train, y_train)
    acc_train = round(clf.score(X_train, y_train), 2)
    acc_test = round(clf.score(X_test, y_test), 2)

    print(i)
    
    max_depth_storage_train.append(i)
    acc_storage_train.append(acc_train)
    
    max_depth_storage_test.append(i)
    acc_storage_test.append(acc_test)


plot_data.PlotData(max_depth_storage_train, acc_storage_train, max_depth_storage_test, acc_storage_test)