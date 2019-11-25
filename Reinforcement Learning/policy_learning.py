# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:45:32 2019

@author: Ray
"""

"""
Solving FrozenLake8x8 environment using Policy iteration.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
import matplotlib.pyplot as plt
import time

def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    v = np.zeros(env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    max_iterations = 200000
    gamma = 1.0
    policies = []
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        policies.append(old_policy_v)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    
    policies.append(new_policy)
    return policy, policies


if __name__ == '__main__':
    ########################################################################################################
    ### Frozen Lake 4x4
    env_name  = 'FrozenLake-v0'
    env = gym.make(env_name)
    optimal_policy, policies = policy_iteration(env, gamma = 1.0)
    
    time_spent = []
    for i in range(1, 1000, 10):
        print(i)
        start = time.time()
        optimal_policy, policies = policy_iteration(env, gamma = 1.0)
        end = time.time()
        time_spent.append(end - start)
    #### Plot the data
    fig_1 = plt.figure()
    ax_1 = plt.axes()
    plt.title('Policy Iteration for FrozenLake 4x4')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Time Spent (ms)')
    plt.plot(range(1, len(time_spent) + 1), time_spent)
    
    scores = []
    for policy in policies:
        scores.append(evaluate_policy(env, policy, gamma = 1.0))
        
    #### Plot the data
    fig_1 = plt.figure()
    ax_1 = plt.axes()
    plt.title('Policy Iteration for FrozenLake 4x4')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Score')
    plt.plot(range(1, len(scores) + 1), scores)
    
    ########################################################################################################
    ### Frozen Lake 16x32
    desc = ["SFFFFFFFFFFFFFHFFFFHHHFFFHHHFFFF",
            "FFFFFFFFHFFFHFFFFFFFFHHFFFHHHHHF",
            "FFFHFFFFHHFHHFFFFFFFFFHHHHFFFFFH",
            "FFFFFHFFFFHFFHFFFFFHHHFFHHFFHFHF",
            "FFFHFFFFHFHFHFFFFFHHFHFHFFHFFFFF",
            "FHHFFFHFFHFHFHFFHHHHHFFFHHHFFFFF",
            "FHFFHFHFFFHHFHHHHHFHFHFFFFFFHHFF",
            "FFFFFFFFHFFFHFFFFFFFFHHFFFHHHHHF",
            "FFFHFFFFHHFHHFFFFFFFFFHHHHFFFFFH",
            "FFFFFHFFFFHFFHFFFFFHHHFFHHFFHFHF",
            "FFFHFFFFHFHFHFFFFFHHFHFHFFHFFFFF",
            "FHHFFFHFFHFHFHFFHHHHHFFFHHHFFFFF",
            "FHFFHFHFFFHHFHHHHHFHFHFFFFFFHHFF",
            "FFFFFFFFHFFFHFFFFFFFFHHFFFHHHHHF",
            "FFFHFFFFHHFHHFFFFFFFFFHHHHFFFFFH",
            "FFFHFFFFFFHHFHFGFHFHFFFFHHHFFFFF"]
    env_name  = 'FrozenLake-v0'
    env = gym.make(env_name, desc = desc)
    optimal_policy, policies = policy_iteration(env, gamma = 1.0)
    
    time_spent = []
    for i in range(1, 500, 10):
        print(i)
        start = time.time()
        optimal_policy, policies = policy_iteration(env, gamma = 1.0)
        end = time.time()
        time_spent.append(end - start)
    #### Plot the data
    fig_1 = plt.figure()
    ax_1 = plt.axes()
    plt.title('Policy Iteration for FrozenLake 16x32')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Time Spent (ms)')
    plt.plot(range(1, len(time_spent) + 1), time_spent)
    
    scores = []
    for policy in policies:
        scores.append(evaluate_policy(env, policy, gamma = 1.0))
        
    #### Plot the data
    fig_1 = plt.figure()
    ax_1 = plt.axes()
    plt.title('Policy Iteration for FrozenLake 16x32')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Score')
    plt.plot(range(1, len(scores) + 1), scores)