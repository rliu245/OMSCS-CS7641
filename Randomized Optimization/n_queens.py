# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:51:06 2019

@author: Ray
"""

import mlrose
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from algorithms import algorithm

def computeNQueens(n = 8):
    num_queens = n
    
    # Define alternative N-Queens fitness function for maximization problem
    def queens_max(state):
        # Initialize counter
        fitness_cnt = 0
        # For all pairs of queens
        for i in range(len(state) - 1):
                for j in range(i + 1, len(state)):
                    # Check for horizontal, diagonal-up and diagonal-down attacks
                    if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):
                        # If no attacks, then increment counter
                        fitness_cnt += 1
        return fitness_cnt
    # Initialize custom fitness function object
    fitness_cust = mlrose.CustomFitness(queens_max)
    
    problem = mlrose.DiscreteOpt(length = num_queens, fitness_fn = fitness_cust, maximize = True, max_val = num_queens)
    
    best_state, best_fitness, score_curves, runtime = algorithm(problem = problem, max_attempts = 500, max_iters = 100)
    
    
    def plot_solution(algorithm_type: str, solution: list) -> None:
        """Given a solution, plot it and save the result to disk."""
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim((0, num_queens))
        ax.set_ylim((0, num_queens))
    
        count = 0
        for queen in solution:
            ax.add_patch(patches.Rectangle((queen, count), 1, 1))
            count += 1
        fig.savefig(algorithm_type + '_' + ''.join([str(a) for a in solution]) + '.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    for key, value in best_state.items():
        plot_solution(key, value)
    
    
    def PlotData(x, y, x2, y2, x3, y3, x4, y4, alpha = 0.5):
        plt.figure()
        plt.plot(x, y, 'r', x2, y2, 'g', x3, y3, 'b', x4, y4, 'c')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        
        red_patch = patches.Patch(color='red', label='Genetic Algorithms')
        green_patch = patches.Patch(color='green', label = 'Simulated Annealing')
        blue_patch = patches.Patch(color='blue', label = 'MIMIC')
        cyan_patch = patches.Patch(color = 'cyan', label = 'Randomized Hill Climb')
        plt.legend(handles=[red_patch, green_patch, blue_patch, cyan_patch])
    
        plt.show()
        
    PlotData(list(range(len(score_curves['genetic algorithm']))), 
             score_curves['genetic algorithm'], 
             list(range(len(score_curves['simulated annealing']))), 
             score_curves['simulated annealing'], 
             list(range(len(score_curves['mimic']))), 
             score_curves['mimic'],
             list(range(len(score_curves['randomized hill climbing']))), 
             score_curves['randomized hill climbing'])
    
    return best_state, best_fitness, score_curves, runtime

def TravelingSalesman():
    # Create list of city coordinates
    """
    coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3), (10, 10), (15, 22), (2, 7), (19, 15), (11, 13),
                    (33, 24), (12, 17), (29, 2), (2, 5), (5, 19), (11, 36), (21, 37), (57, 22), (36, 12), (19, 20), (13, 19), (13, 54), (0, 5),
                    (44, 14), (45, 45), (23, 20), (16, 2), (3, 29), (21, 59), (18, 29), (2, 2), (19, 17), (39, 14), (9, 9), (48, 14), (59, 59), (29, 1)]
    """
    weights = [10, 5, 2, 8, 15, 20, 5, 2, 1, 20, 8, 6, 14, 22, 50, 5, 10, 12, 12, 18, 26, 32, 4, 8, 10, 5, 22,
               10, 5, 2, 8, 15, 20, 5, 2, 1, 20, 8, 6, 14, 22, 50, 5, 10, 12, 12, 18, 26, 32, 4, 8, 10, 5, 22,
               10, 5, 2, 8, 15, 20, 5, 2, 1, 20, 8, 6, 14, 22, 50, 5, 10, 12, 12, 18, 26, 32, 4, 8, 10, 5, 22]
    values = [1, 2, 3, 4, 5, 2, 5, 10, 1, 4, 10, 2, 2, 8, 100, 5, 15, 24, 8, 14, 36, 10, 5, 2, 120, 4, 8,
              1, 2, 3, 4, 5, 2, 5, 10, 1, 4, 10, 2, 2, 8, 100, 5, 15, 24, 8, 14, 36, 10, 5, 2, 120, 4, 8,
              1, 2, 3, 4, 5, 2, 5, 10, 1, 4, 10, 2, 2, 8, 100, 5, 15, 24, 8, 14, 36, 10, 5, 2, 120, 4, 8]
    max_weight_pct = 0.6
    
    #n = len(coords_list)
    n = len(weights)
    
    # Initialize fitness function object using coords_list
    #fitness_coords = mlrose.TravellingSales(coords = coords_list)
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    
    # Define optimization problem object
    problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = True)
    
    best_state, best_fitness, score_curves, runtime = algorithm(problem = problem, max_attempts = 500, max_iters = 100)
    
    def plot_solution(algorithm_type: str, solution: list) -> None:
        """Given a solution, plot it and save the result to disk."""
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim((0, n))
        ax.set_ylim((0, n))
    
        count = 0
        for queen in solution:
            ax.add_patch(patches.Rectangle((queen, count), 1, 1))
            count += 1
        fig.savefig(algorithm_type + '_' + ''.join([str(a) for a in solution]) + '.png', dpi = 150, bbox_inches = 'tight')
        plt.close(fig)
    
    for key, value in best_state.items():
        plot_solution(key, value)
    
    
    def PlotData(x, y, x2, y2, x3, y3, x4, y4):
        plt.figure()
        plt.plot(x, y, 'r', x2, y2, 'g', x3, y3, 'b', x4, y4, 'c', alpha = 0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        
        red_patch = patches.Patch(color='red', label='Genetic Algorithms')
        green_patch = patches.Patch(color='green', label = 'Simulated Annealing')
        blue_patch = patches.Patch(color='blue', label = 'MIMIC')
        cyan_patch = patches.Patch(color = 'cyan', label = 'Randomized Hill Climb')
        plt.legend(handles=[red_patch, green_patch, blue_patch, cyan_patch])
    
        plt.show()
        
    PlotData(list(range(len(score_curves['genetic algorithm']))), 
             score_curves['genetic algorithm'], 
             list(range(len(score_curves['simulated annealing']))), 
             score_curves['simulated annealing'], 
             list(range(len(score_curves['mimic']))), 
             score_curves['mimic'],
             list(range(len(score_curves['randomized hill climbing']))), 
             score_curves['randomized hill climbing'])
    
    return best_state, best_fitness, score_curves, runtime
    
def MaxKColoring():
    # edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
    state = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    n = len(state)
    
    #fitness = mlrose.MaxKColor(edges)
    fitness = mlrose.FourPeaks(t_pct=0.15)

    # Define optimization problem object
    #problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = False, max_val = n)
    problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = True, max_val = 2)
    
    best_state, best_fitness, score_curves, runtime = algorithm(problem = problem, max_attempts = 500, max_iters = 50)
    
    
    def plot_solution(algorithm_type: str, solution: list) -> None:
        """Given a solution, plot it and save the result to disk."""
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim((0, n))
        ax.set_ylim((0, n))
    
        count = 0
        for queen in solution:
            ax.add_patch(patches.Rectangle((queen, count), 1, 1))
            count += 1
        fig.savefig(algorithm_type + '_' + ''.join([str(a) for a in solution]) + '.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    for key, value in best_state.items():
        plot_solution(key, value)
    
    
    def PlotData(x, y, x2, y2, x3, y3, x4, y4):
        plt.figure()
        plt.plot(x, y, 'r', x2, y2, 'g', x3, y3, 'b', x4, y4, 'c', alpha = 0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        
        plt.xticks(np.arange(0, 51, 3.0))
        #plt.yticks(np.arange(-5, 5, 1))
        
        red_patch = patches.Patch(color='red', label='Genetic Algorithms')
        green_patch = patches.Patch(color='green', label = 'Simulated Annealing')
        blue_patch = patches.Patch(color='blue', label = 'MIMIC')
        cyan_patch = patches.Patch(color = 'cyan', label = 'Randomized Hill Climb')
        plt.legend(handles=[red_patch, green_patch, blue_patch, cyan_patch])
        
        plt.show()
        
    PlotData(list(range(len(score_curves['genetic algorithm']))), 
             score_curves['genetic algorithm'], 
             list(range(len(score_curves['simulated annealing']))), 
             score_curves['simulated annealing'], 
             list(range(len(score_curves['mimic']))), 
             score_curves['mimic'],
             list(range(len(score_curves['randomized hill climbing']))), 
             score_curves['randomized hill climbing'])
    
    return best_state, best_fitness, score_curves, runtime

if __name__ == "__main__":
    best_state_q, best_fitness_q, score_curves_q, runtime_q = computeNQueens(100)
    best_state, best_fitness, score_curves, runtime = TravelingSalesman()
    best_state_c, best_fitness_c, score_curves_c, runtime_c = MaxKColoring()














