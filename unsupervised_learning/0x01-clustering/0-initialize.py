#!/usr/bin/env python3
""" Clusters centroids init """


import numpy as np


def initialize(X, k):
    """ cluster centroid init function """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    n, d = X.shape
    if not isinstance(k, int) or k <= 0 or k > n:
        return None
    cen = np.random.uniform(low=np.min(X, axis=0),
                           high=np.max(X, axis=0),
                           size=(k, d))
    return cen
