# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:31:52 2019

@author: Ray
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def SvmPlot(X, y, X_test, y_test):
    h = .02  # step size in the mesh
    
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C, max_iter = 10000).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C, max_iter = 10000).fit(X, y)
    sigmoid_svc = svm.SVC(kernel='sigmoid', gamma=0.7, C=C, max_iter = 10000).fit(X, y)
    poly_svc2 = svm.SVC(kernel='poly', degree=2, C=C, max_iter = 10000).fit(X, y)
    poly_svc3 = svm.SVC(kernel='poly', degree=3, C=C, max_iter = 10000).fit(X, y)
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # title for the plots
    labels = ['Linear',
              'Sigmoid',
              'RBF',
              'Polynomial 2',
              'Polynomial 3']

    training = [round(svc.score(X, y), 2), round(rbf_svc.score(X, y), 2), round(sigmoid_svc.score(X, y), 2), 
                 round(poly_svc2.score(X, y), 2), round(poly_svc3.score(X, y), 2)]
    test = [round(svc.score(X_test, y_test), 2), round(rbf_svc.score(X_test, y_test), 2), round(sigmoid_svc.score(X_test, y_test), 2), 
                 round(poly_svc2.score(X_test, y_test), 2), round(poly_svc3.score(X_test, y_test), 2)]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, training, width, label='Train')
    rects2 = ax.bar(x + width/2, test, width, label='Test')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Support Vector Classifier: Accuracy Metric Using Different Kernels')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features. We could
                          # avoid this ugly slicing by using a two-dim dataset
    y = iris.target
    
    SvmPlot(X, y, X, y)
    
    