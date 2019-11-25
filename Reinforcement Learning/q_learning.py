# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:59:45 2019

@author: Ray
"""


# -*- coding: utf-8 -*-
# ref: https://gym.openai.com/evaluations/eval_1lfzNKEHS9GA7nNWE73w

import numpy as np
import gym
import time

# Q learning params
ALPHA = 0.1 # learning rate
GAMMA = 0.99 # reward discount
LEARNING_COUNT = 100000
TEST_COUNT = 10000

TURN_LIMIT = 100

class Agent:
    def __init__(self, env):
        self.env = env
        self.episode_reward = 0.0
        self.q_val = np.zeros(16 * 4).reshape(16, 4).astype(np.float32)

    def learn(self):
        # one episode learning
        state = self.env.reset()
        #self.env.render()
        
        for t in range(TURN_LIMIT):
            act = self.env.action_space.sample() # random
            next_state, reward, done, info = self.env.step(act)
            q_next_max = np.max(self.q_val[next_state % 15])
            # Q <- Q + a(Q' - Q)
            # <=> Q <- (1-a)Q + a(Q')
            self.q_val[state % 15][act] = (1 - ALPHA) * self.q_val[state % 15][act]\
                                 + ALPHA * (reward + GAMMA * q_next_max)
            
            #self.env.render()
            if done:
                return reward
            else:
                state = next_state

    def test(self):
        state = self.env.reset()
        for t in range(TURN_LIMIT):
            act = np.argmax(self.q_val[state % 15])
            next_state, reward, done, info = self.env.step(act)
            if done:
                return reward
            else:
                state = next_state
        return 0.0 # over limit

def main():
    ########################################################################################################
    ### Frozen Lake 4x4
    env = gym.make("FrozenLake-v0")
    agent = Agent(env)

    print("###### LEARNING #####")
    reward_total = 0.0
    start = time.time()
    for i in range(LEARNING_COUNT):
        reward_total += agent.learn()
    end = time.time()
    print("episodes      : {}".format(LEARNING_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / LEARNING_COUNT))
    print("Q Value       :\n{}".format(agent.q_val))
    print("Time Spent    : {:.2f}".format(end - start))
    print()

    print("###### TEST #####")
    reward_total = 0.0
    for i in range(TEST_COUNT):
        reward_total += agent.test()
    print("episodes      : {}".format(TEST_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / TEST_COUNT))
    
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
    env = gym.make("FrozenLake-v0", desc = desc)
    agent = Agent(env)

    print("###### LEARNING #####")
    reward_total = 0.0
    start = time.time()
    for i in range(LEARNING_COUNT):
        reward_total += agent.learn()
    end = time.time()
    print("episodes      : {}".format(LEARNING_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / LEARNING_COUNT))
    print("Q Value       :\n{}".format(agent.q_val))
    print("Time Spent    : {:.2f}".format(end - start))
    print()

    print("###### TEST #####")
    reward_total = 0.0
    for i in range(TEST_COUNT):
        reward_total += agent.test()
    print("episodes      : {}".format(TEST_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / TEST_COUNT))

if __name__ == "__main__":
    main()