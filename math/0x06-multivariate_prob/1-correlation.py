#!/usr/bin/env python3
""" Correlation matrix """
import numpy as np


def correlation(C):
    """ correlation matrix """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    var = np.sqrt(np.diag(C))
    var = np.expand_dims(var, 0)
    o_v = np.outer(var, var)
    corr = C / o_v
    corr[C == 0] = 0
    return corr
