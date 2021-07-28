#!/usr/bin/env python3
""" Q affinities """
import numpy as np


def Q_affinities(Y):
    """ Q affinities function """
    n, ndim = Y.shape
    Q = np.zeros((n, n))
    sum_Y = np.sum(np.square(Y), 1, keepdims=True)
    D = sum_Y + sum_Y.T - 2*np.dot(Y, Y.T)
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return(Q, num)
