#!/usr/bin/env python3
""" Mean & covariance """
import numpy as np


def mean_cov(X):
    """ mean & covariance """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, 0)
    mean = np.expand_dims(mean, 0)
    X -= mean

    cov = np.matmul(X.T, X)/(n - 1)
    return mean, cov
