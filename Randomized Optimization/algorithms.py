# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:37:43 2019

@author: Ray
"""
import time
import mlrose

def algorithm(problem, max_attempts = 100, max_iters = 1000):
    runtime = {'simulated annealing': 0,
               'genetic algorithm': 0,
               'mimic': 0,
               'randomized hill climbing': 0 }
    
    best_state = {}
    best_fitness = {}
    score_curves = {}
    
    # Define decay schedule
    #schedule = mlrose.ExpDecay()
    schedule = mlrose.GeomDecay()
    
    ### Compute Genetic Algorithm
    start = time.time()
    # Solve problem using genetic algorithm
    best_state_ga, best_fitness_ga, curve_ga = mlrose.genetic_alg(problem, pop_size = 500, mutation_prob = 0.001,
                                                          max_attempts = max_attempts, max_iters = max_iters,
                                                          curve = True, random_state = 1)
    end = time.time()
    runtime['genetic algorithm'] = end - start
    best_state['genetic algorithm'] = best_state_ga
    best_fitness['genetic algorithm'] = best_fitness_ga
    score_curves['genetic algorithm'] = curve_ga
    
    ### Compute simulated annealing
    start = time.time()
    # Solve problem using simulated annealing
    best_state_sa, best_fitness_sa, curve_sa = mlrose.simulated_annealing(problem, schedule = schedule,
                                                          max_attempts = max_attempts, max_iters = max_iters,
                                                          curve = True, random_state = 1)
    end = time.time()
    runtime['simulated annealing'] = end - start
    best_state['simulated annealing'] = best_state_sa
    best_fitness['simulated annealing'] = best_fitness_sa
    score_curves['simulated annealing'] = curve_sa
    
    ### Compute MIMIC
    start = time.time()
    # Solve problem using MIMIC
    best_state_mimic, best_fitness_mimic, curve_mimic = mlrose.mimic(problem, pop_size = 500, keep_pct = 0.8,
                                                          max_attempts = max_attempts, max_iters = max_iters,
                                                          curve = True, random_state = 1)
    end = time.time()
    runtime['mimic'] = end - start
    best_state['mimic'] = best_state_mimic
    best_fitness['mimic'] = best_fitness_mimic
    score_curves['mimic'] = curve_mimic
    
    ### Compute Randomized Hill Climbing
    start = time.time()
    # Solve problem using Randomized Hill Climbing
    best_state_rhc, best_fitness_rhc, curve_rhc = mlrose.random_hill_climb(problem,
                                                          max_attempts = max_attempts, max_iters = max_iters,
                                                          curve = True, random_state = 1)
    end = time.time()
    runtime['randomized hill climbing'] = end - start
    best_state['randomized hill climbing'] = best_state_rhc
    best_fitness['randomized hill climbing'] = best_fitness_rhc
    score_curves['randomized hill climbing'] = curve_rhc
    
    return best_state, best_fitness, score_curves, runtime
    