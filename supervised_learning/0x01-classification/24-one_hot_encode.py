#!/usr/bin/env python3
""" One-hot encoding """
import numpy as np


def one_hot_encode(Y, classes):
    """ one hot encoding method """
    if type(Y) != np.ndarray or len(Y) < 1:
        return None
    if type(classes) != int or classes <= np.amax(Y):
        return None

    one_hot_matrix = np.zeros((classes, Y.shape[0]))
    rows = np.arange(Y.size)
    one_hot_matrix[Y, rows] = 1

    return one_hot_matrix
