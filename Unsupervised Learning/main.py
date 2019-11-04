# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 19:55:59 2019

@author: Ray
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import preprocess

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis

from sklearn.random_projection import GaussianRandomProjection

# feature importance using ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier

# neural network for accuracy comparison
from sklearn.neural_network import MLPClassifier

seed = 50

def run_kmeans(data):
    scores = []
    for i in range(1, 30):
        print(f'KMeans: with K = {i}')
        clf = KMeans(n_clusters = i, random_state = seed, precompute_distances = True)
        clf.fit(data)
        scores.append(clf.inertia_)
    
    return scores

def run_pca_kmeans(data, principal_components = 1):
    clf = PCA(n_components = principal_components)
    data = clf.fit_transform(data)
        
    scores = []
    for i in range(1, 30):
        print(f'KMeans and PCA: with K = {i}')
        clf = KMeans(n_clusters = i, random_state = seed, precompute_distances = True)
        clf.fit(data)
        scores.append(clf.inertia_)
    
    return scores

def run_ica_kmeans(data, principal_components = 1):
    clf = FastICA(n_components = principal_components)
    data = clf.fit_transform(data)
        
    scores = []
    for i in range(1, 30):
        print(f'KMeans and ICA: with K = {i}')
        clf = KMeans(n_clusters = i, random_state = seed, precompute_distances = True)
        clf.fit(data)
        scores.append(clf.inertia_)
    
    return scores

def run_rp_kmeans(data, k = 1):    
    scores = []
    for i in range(1, len(data.iloc[0])):
        print(f'KMeans and Random Projection: with num_components = {i}')
        clf_kmeans = KMeans(n_clusters = k, random_state = seed, precompute_distances = True)
        new_data = run_random_projection(data, num_components = i)
        clf_kmeans.fit(new_data)
        scores.append(clf_kmeans.inertia_)
    
    return scores
    
def run_EM(data):
    scores = []
    for i in range(1, len(data.iloc[0]) * 2):
        print(f'EM: with n_components = {i}')
        clf = GaussianMixture(n_components = i)
        clf.fit(data)
        scores.append(clf.score(data))
    
    return scores

def run_pca_EM(data, principal_components = 1):
    clf = PCA(n_components = principal_components)
    data = clf.fit_transform(data)
        
    scores = []
    for i in range(1, len(data[0]) * 2):
        print(f'EM and PCA: with K = {i}')
        clf = GaussianMixture(n_components = i)
        clf.fit(data)
        scores.append(clf.score(data))
    
    return scores

def run_ica_EM(data, principal_components = 1):
    clf = FastICA(n_components = principal_components)
    data = clf.fit_transform(data)
        
    scores = []
    for i in range(1, len(data[0]) * 2):
        print(f'EM and ICA: with K = {i}')
        clf = GaussianMixture(n_components = i)
        clf.fit(data)
        scores.append(clf.score(data))
    
    return scores

def run_rp_EM(data, k = 1):    
    scores = []
    for i in range(1, len(data.iloc[0])):
        print(f'EM and Random Projection: with num_components = {i}')
        clf_EM = GaussianMixture(n_components = k)
        new_data = run_random_projection(data, num_components = i)
        clf_EM.fit(new_data)
        scores.append(clf_EM.score(new_data))
    
    return scores

def run_pca(data):
    scores = []
    for i in range(1, len(data.iloc[0])):
        print(f'PCA: with n_components = {i}')
        clf = PCA(n_components = i)
        clf.fit(data)
        scores.append(clf.score(data))
    
    return scores

def run_ica(data):
    scores = []
    for i in range(1, len(data.iloc[0])):
        print(f'ICA: with n_components = {i}')
        clf = FastICA(n_components = i, random_state = seed)
        temp = clf.fit_transform(data)
        scores.append(kurtosis(temp, axis = None))
        
    return scores

def run_random_projection(data, num_components):
    clf = GaussianRandomProjection(n_components = num_components, random_state = seed)
    return clf.fit_transform(data)


def run_feature_importance(data, y, title):
    column_names = data.columns
    
    forest = ExtraTreesClassifier(n_estimators = 250, random_state = seed)
    forest.fit(data, y)
    
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(data.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title(title)
    plt.bar(range(data.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    # plt.xticks(range(data.shape[1]), indices)
    plt.xticks(range(data.shape[1]), [column_names[index] for index in indices])
    plt.xlim([-1, data.shape[1]])
    plt.show()
    
    return [column_names[indices[i]] for i in range(0, 3)]

def run_nn(X, y):
    clf = MLPClassifier(solver = 'sgd', activation = 'relu', learning_rate = 'adaptive', hidden_layer_sizes = (2, 8), 
                        batch_size = 32, shuffle = True, early_stopping = True, max_iter = 1000, random_state = 1).fit(X, y)
    
    return clf.score(X, y)

def compute_neural_network(a, b):
    ### PCA
    ### ICA
    ### RP
    ### Feature Importance
    
    # dataset a
    clf = PCA(n_components = 3)
    temp = clf.fit_transform(a[0])
    print(f'Admissions Dataset: PCA {run_nn(temp, a[1])}')
        
    clf = FastICA(n_components = 7, random_state = seed)
    temp = clf.fit_transform(a[0])
    print(f'Admissions Dataset: ICA {run_nn(temp, a[1])}')
    
    clf = GaussianRandomProjection(n_components = 2, random_state = seed)
    temp = clf.fit_transform(a[0])
    print(f'Admissions Dataset: RP {run_nn(temp, a[1])}')
    
    important_features = ['CGPA', 'GRE Score', 'TOEFL Score']
    temp_data = dict()
    for feature in important_features:
        temp_data[feature] = a[0][feature]
    
    temp = pd.DataFrame(temp_data)
    
    print(f'Admissions Dataset: Feature Importance {run_nn(temp, a[1])}')
    
    # dataset b
    ## 5 PCA, 7 ICA
    clf = PCA(n_components = 5)
    temp = clf.fit_transform(b[0])
    print(f'Income Dataset: PCA {run_nn(temp, b[1])}')
        
    clf = FastICA(n_components = 7, random_state = seed)
    temp = clf.fit_transform(b[0])
    print(f'Income Dataset: ICA {run_nn(temp, b[1])}')
    
    clf = GaussianRandomProjection(n_components = 2, random_state = seed)
    temp = clf.fit_transform(b[0])
    print(f'Income Dataset Dataset: RP {run_nn(temp, b[1])}')
    
    important_features = ['fnlwgt', 'age', 'education-num']
    temp_data = dict()
    for feature in important_features:
        temp_data[feature] = b[0][feature]
    
    temp = pd.DataFrame(temp_data)
    
    print(f'Income Dataset: Feature Importance {run_nn(temp, b[1])}')

if __name__ == "__main__":
    plt.style.use('seaborn-whitegrid')
    
    ######## Dataset 1 Preprocessing
    a = pd.read_csv('Admission_Predict.csv')
    for column in a.columns:
        indexNames = a[a[column] == ' ?'].index
        # Delete these row indexes from dataFrame
        a.drop(indexNames, inplace=True)
    a, a_y = preprocess.PreProcessAdmission(a.drop('Chance of Admit', axis = 1), a['Chance of Admit'])
    
    ######## Dataset 2 Preprocessing
    b = pd.read_csv('adult.test')
    for column in b.columns:
        indexNames = b[b[column] == ' ?'].index
        # Delete these row indexes from dataFrame
        b.drop(indexNames, inplace=True)
    b, b_y = preprocess.PreProcessIncome(b.drop('income', axis = 1), b['income'])

    dataset = [a, b]
    # Run the clustering algorithms on the datasets and describe what you see.
    ############################# K-Means
    i = 0
    for data in dataset:
        fig_1 = plt.figure()
        ax_1 = plt.axes()
        scores = run_kmeans(data)
        scores.insert(0, scores[0])
        if i == 0:
            plt.title('K-Means for Admissions Dataset')
            i += 1
        else:
            plt.title('K-Means for Income Dataset')
        plt.xlabel('K')
        plt.ylabel('WCSS')
        plt.plot(range(1, len(scores) + 1), scores)
    
    ############################# EM Algorithm
    i = 0
    for data in dataset:
        fig_1 = plt.figure()
        ax_1 = plt.axes()
        scores = run_EM(data)
        scores.insert(0, scores[0])
        if i == 0:
            plt.title('EM for Admissions Dataset')
            i += 1
        else:
            plt.title('EM for Income Dataset')
        plt.xlabel('Number of Components')
        plt.ylabel('Log-Likelihood loss')
        plt.plot(range(1, len(scores) + 1), scores)
    
    # Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
    # Evaluation metric for PCA: Kaiser Criterion: use PCA with eigenvalues > 1
    ############################# PCA
    i = 0
    for data in dataset:
        fig_1 = plt.figure()
        ax_1 = plt.axes()
        scores = run_pca(data)
        scores.insert(0, scores[0])
        if i == 0:
            plt.title('PCA for Admissions Dataset')
            i += 1
        else:
            plt.title('PCA for Income Dataset')
        plt.xlabel('Number of Components')
        plt.ylabel('Log-Likelihood loss')
        plt.plot(range(1, len(scores) + 1), scores)
    
    ### 3 Principal Components for Admissions Dataset
    ### 5 Principal Components for Income Dataset
    ############################# PCA + K-Means
    i = 0
    for data in dataset:
        fig_1 = plt.figure()
        ax_1 = plt.axes()
        if i == 0:
            scores = run_pca_kmeans(data, principal_components = 3)
            plt.title("PCA(3 PC's) and K-Means for Admissions Dataset")
            i += 1
        else:
            scores = run_pca_kmeans(data, principal_components = 5)
            plt.title("PCA(5 PC's) and K-Means for Income Dataset")
        plt.xlabel('K')
        plt.ylabel('WCSS')
        plt.plot(range(1, len(scores) + 1), scores)
    
    ### 3 Principal Components for Admissions Dataset
    ### 5 Principal Components for Income Dataset
    ############################# PCA + EM Algorithm
    i = 0
    for data in dataset:
        fig_1 = plt.figure()
        ax_1 = plt.axes()
        if i == 0:
            scores = run_pca_EM(data, principal_components = 3)
            plt.title("PCA(3 PC's) and EM for Admissions Dataset")
            i += 1
        else:
            scores = run_pca_EM(data, principal_components = 5)
            plt.title("PCA(5 PC's) and EM for Income Dataset")
        plt.xlabel('Number of Components')
        plt.ylabel('Log-Likelihood loss')
        plt.plot(range(1, len(scores) + 1), scores)
    
    ############################# ICA
    i = 0
    for data in dataset:
        fig_1 = plt.figure()
        ax_1 = plt.axes()
        scores = run_ica(data)
        scores.insert(0, scores[0])
        if i == 0:
            plt.title('ICA for Admissions Dataset')
            i += 1
        else:
            plt.title('ICA for Income Dataset')
        plt.xlabel('Number of Components')
        plt.ylabel('Kurtosis Score')
        plt.plot(range(1, len(scores) + 1), scores)
        
    ### 7 Principal Components for Admissions Dataset
    ### 7 Principal Components for Income Dataset
    ############################# ICA + K-Means
    i = 0
    for data in dataset:
        fig_1 = plt.figure()
        ax_1 = plt.axes()
        if i == 0:
            scores = run_ica_kmeans(data, principal_components = 7)
            plt.title("ICA(7 Components) and K-Means for Admissions Dataset")
            i += 1
        else:
            scores = run_pca_kmeans(data, principal_components = 7)
            plt.title("ICA(7 Components) and K-Means for Income Dataset")
        plt.xlabel('K')
        plt.ylabel('WCSS')
        plt.plot(range(1, len(scores) + 1), scores)
        
    ### 7 Principal Components for Admissions Dataset
    ### 7 Principal Components for Income Dataset
    ############################# ICA + EM Algorithm
    i = 0
    for data in dataset:
        fig_1 = plt.figure()
        ax_1 = plt.axes()
        if i == 0:
            scores = run_ica_EM(data, principal_components = 7)
            plt.title("ICA(3 PC's) and EM for Admissions Dataset")
            i += 1
        else:
            scores = run_ica_EM(data, principal_components = 7)
            plt.title("ICA(5 PC's) and EM for Income Dataset")
        plt.xlabel('Number of Components')
        plt.ylabel('Log-Likelihood loss')
        plt.plot(range(1, len(scores) + 1), scores)
        
    ############################# Randomized Projections
    i = 0
    for data in dataset:
        fig_1 = plt.figure()
        ax_1 = plt.axes()
        scores = run_rp_kmeans(data, k = 5)
        scores.insert(0, scores[0])
        if i == 0:
            plt.title('Randomized Projection for Admissions Dataset')
            i += 1
        else:
            plt.title('Randomized Projection for Income Dataset')
        plt.xlabel('Number of Components')
        plt.ylabel('WCSS')
        plt.plot(range(1, len(scores) + 1), scores)
    
    ############################# Feature Importance (Feature Selection Algorithm of my Choice)
    i = 0
    for data in dataset:
        if i == 0:
            important_features = run_feature_importance(data, a_y, 'Feature Importance for Admissions Dataset')
        else:
            important_features = run_feature_importance(data, b_y, 'Feature Importance for Income Dataset')
        
        temp_data = dict()
        for feature in important_features:
            temp_data[feature] = data[feature]
        
        temp_data = pd.DataFrame(temp_data)
        fig_1 = plt.figure()
        ax_1 = plt.axes()
        scores = run_kmeans(temp_data)
        scores.insert(0, scores[0])
        if i == 0:
            plt.title('Feature Importance(Top 3) and K-Means for Admissions Dataset')
            i += 1
        else:
            plt.title('Feature Importance(Top 3) and K-Means for Income Dataset')
        plt.xlabel('K')
        plt.ylabel('WCSS')
        plt.plot(range(1, len(scores) + 1), scores)
    
    compute_neural_network((a, a_y), (b, b_y))
    
    
        
        
        
        
        
        
