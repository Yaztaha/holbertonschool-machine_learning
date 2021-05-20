#!/usr/bin/env python3
""" Shuffle data points """
import numpy as np


def shuffle_data(X, Y):
    """ function of shuffling data """
    shuffle = np.random.permutation(X.shape[0])
    shuf_X = X[shuffle]
    shuf_Y = Y[shuffle]
    return shuf_X, shuf_Y
