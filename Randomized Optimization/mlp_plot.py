print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import mlrose
 
"""
def PlotData(x, y, x2, y2, x3, y3):
        plt.figure()
        plt.plot(x, y, 'r', x2, y2, 'g', x3, y3, 'b', alpha = 0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Errors')
        
        #plt.xticks(np.arange(0, 1001, 32.0))
        
        red_patch = patches.Patch(color='red', label='Randomized Hill Climbing')
        green_patch = patches.Patch(color='green', label = 'Simulated Annealing')
        blue_patch = patches.Patch(color='blue', label = 'Genetic Algorithm')
        plt.legend(handles=[red_patch, green_patch, blue_patch])
        
        plt.show()
"""

def PlotData(x, y, x2, y2, title = 'Test'):
        plt.figure()
        plt.plot(x, y, 'r', x2, y2, 'g', alpha = 0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Errors')
        
        #plt.xticks(np.arange(0, 1001, 32.0))
        
        red_patch = patches.Patch(color='red', label='Train')
        green_patch = patches.Patch(color='green', label='Test')
        plt.legend(handles=[red_patch, green_patch])
        #plt.set_title(title)
        plt.show()

def bar_plot(X, y, train_errors, test_errors):
    for i in range(len(train_errors)):
        train_errors[i] = round(train_errors[i], 3)
        test_errors[i] = round(test_errors[i], 3)
    
    h = .02  # step size in the mesh
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # title for the plots
    labels = ['0',
              '1',
              '2',
              '3',
              '4',
              '5']

    training = train_errors
    test = test_errors
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, training, width, label='Train')
    rects2 = ax.bar(x + width/2, test, width, label='Test')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Number of Restarts')
    ax.set_ylabel('Error')
    ax.set_title('Error vs Number of Restarts')
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

def MLP_rhc(X, y, X_test, y_test):
    print('Running Randomized Hill Climbing')
    # Perform Random Hill Climbing
    train_errors = []
    test_errors = []
    for i in range(0, 6):
        print(i)
        clf_rhc = mlrose.NeuralNetwork(hidden_nodes = [8, 8], activation = 'tanh', is_classifier = True, algorithm = 'random_hill_climb', max_iters = 500, bias = True, 
                                       learning_rate = 0.01, early_stopping = False, clip_max = 100.0, restarts = i, max_attempts = 10, curve = True, 
                                       random_state = 1)
        clf_rhc = clf_rhc.fit(X, y)
        train_errors.append(clf_rhc.loss)
        test_errors.append(clf_rhc.score(X_test, y_test))
    
    clf_rhc = mlrose.NeuralNetwork(hidden_nodes = [8, 8], activation = 'tanh', is_classifier = True, algorithm = 'random_hill_climb', max_iters = 500, bias = True, 
                                       learning_rate = 0.01, early_stopping = False, clip_max = 100.0, restarts = 1, max_attempts = 10, curve = True, 
                                       random_state = 1)
    clf_rhc = clf_rhc.fit(X, y)
    pred_score_rhc = [clf_rhc.score(X_test, y_test)]
    
    PlotData(list(range(len(clf_rhc.fitness_curve))), 
             clf_rhc.fitness_curve, 
             list(range(len(pred_score_rhc))), 
             pred_score_rhc, 'Error vs Iterations')
    
    return clf_rhc, train_errors, test_errors
        
def MLP_sa(X, y, X_test, y_test):
    print('Running Simulated Annealing')
    # Perform Simulated Annealing
    clf_sa = mlrose.NeuralNetwork(hidden_nodes = [8, 8], activation = 'tanh', is_classifier = True, algorithm = 'simulated_annealing', max_iters = 500, bias = True, 
                                  learning_rate = 0.01, early_stopping = False, clip_max = 100.0, schedule = mlrose.decay.GeomDecay(), max_attempts = 10, curve = True, 
                                  random_state = 1)
    
    clf_sa = clf_sa.fit(X, y)
    pred_score_sa = [clf_sa.score(X_test, y_test)]
    
    PlotData(list(range(len(clf_sa.fitness_curve))), 
             clf_sa.fitness_curve, 
             list(range(len(pred_score_sa))), 
             pred_score_sa, 'Error vs Iterations')
    
def MLP_ga(X, y, X_test, y_test):
    print('Running Genetic Algorithm')
    # Perform Genetic Algorithm
    clf_ga = mlrose.NeuralNetwork(hidden_nodes = [8, 8], activation = 'tanh', is_classifier = True, algorithm = 'genetic_alg', max_iters = 500, bias = True, 
                                  learning_rate = 0.01, early_stopping = False, clip_max = 50.0, pop_size = 200, mutation_prob = 0.1, max_attempts = 10, curve = True, 
                                  random_state = 1)
    
    clf_ga = clf_ga.fit(X, y)
    pred_score_ga = [clf_ga.score(X_test, y_test)]

    PlotData(list(range(len(clf_ga.fitness_curve))), 
             clf_ga.fitness_curve, 
             list(range(len(pred_score_ga))), 
             pred_score_ga, 'Error vs Iterations')
    
    
    
    