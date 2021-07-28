#!/usr/bin/env python3
""" Y gradient """
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """ function that calculates gradient of Y """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)

    PQ = P - Q
    dY = np.zeros((n, ndim))

    for i in range(n):
        tiles = np.tile(PQ[:, i] * num[:, i], (ndim, 1))
        dY[i, :] = np.sum(tiles.T * (Y[i, :] - Y), 0)
    return dY, Q
