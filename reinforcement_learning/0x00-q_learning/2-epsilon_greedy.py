#!/usr/bin/env python3
""" Exploration vs Exploitation tradeoff """
import gym
import numpy as np

def epsilon_greedy(Q, state, epsilon):
    """ epsilon greedy function """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        return np.argmax(Q[state, :])
    else:
        return np.random.randint(Q.shape[1])
