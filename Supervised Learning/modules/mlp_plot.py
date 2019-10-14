print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def MultiLayerPerceptronPlot(X, y, X_test, y_test):
    h = .02  # step size in the mesh

    clf_1x4 = MLPClassifier(solver = 'sgd', activation = 'relu', learning_rate = 'adaptive', hidden_layer_sizes = (1, 4), 
                        batch_size = 32, shuffle = True, early_stopping = True, max_iter = 1000, random_state = 1).fit(X, y)
    clf_1x8 = MLPClassifier(solver = 'sgd', activation = 'relu', learning_rate = 'adaptive', hidden_layer_sizes = (1, 8), 
                        batch_size = 32, shuffle = True, early_stopping = True, max_iter = 1000, random_state = 1).fit(X, y)
    clf_1x16 = MLPClassifier(solver = 'sgd', activation = 'relu', learning_rate = 'adaptive', hidden_layer_sizes = (1, 16), 
                        batch_size = 32, shuffle = True, early_stopping = True, max_iter = 1000, random_state = 1).fit(X, y)
    clf_2x4 = MLPClassifier(solver = 'sgd', activation = 'relu', learning_rate = 'adaptive', hidden_layer_sizes = (2, 4), 
                        batch_size = 32, shuffle = True, early_stopping = True, max_iter = 1000, random_state = 1).fit(X, y)
    clf_2x8= MLPClassifier(solver = 'sgd', activation = 'relu', learning_rate = 'adaptive', hidden_layer_sizes = (2, 8), 
                        batch_size = 32, shuffle = True, early_stopping = True, max_iter = 1000, random_state = 1).fit(X, y)
    clf_2x16 = MLPClassifier(solver = 'sgd', activation = 'relu', learning_rate = 'adaptive', hidden_layer_sizes = (2, 16), 
                        batch_size = 32, shuffle = True, early_stopping = True, max_iter = 1000, random_state = 1).fit(X, y)
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # title for the plots
    labels = ['MLP-1x4',
              'MLP-1x8',
              'MLP-1x16',
              'MLP-2x4',
              'MLP-2x8',
              'MLP-2x16']

    training = [round(clf_1x4.score(X, y), 2), round(clf_1x8.score(X, y), 2), round(clf_1x16.score(X, y), 2), 
                 round(clf_2x4.score(X, y), 2), round(clf_2x8.score(X, y), 2), round(clf_2x16.score(X, y), 2)]
    test = [round(clf_1x4.score(X_test, y_test), 2), round(clf_1x8.score(X_test, y_test), 2), round(clf_1x16.score(X_test, y_test), 2), 
                 round(clf_2x4.score(X_test, y_test), 2), round(clf_2x8.score(X_test, y_test), 2), round(clf_2x16.score(X_test, y_test), 2)]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, training, width, label='Train')
    rects2 = ax.bar(x + width/2, test, width, label='Test')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Multi-Layer Perceptron Classifier: Accuracy Metric Using Different Number of Layers and Neurons')
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