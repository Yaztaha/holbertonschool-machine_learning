#!/usr/bin/env python3
""" Variable init """
import numpy as np


def P_init(X, perplexity):
    """ Var init function """
    n, d = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
