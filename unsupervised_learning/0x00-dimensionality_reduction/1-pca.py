#!/usr/bin/env python3
""" Principal component analysis """
import numpy as np


def pca(X, ndim):
    """ Function PCA2 with dim """
    X -= np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X)
    w = vh.T
    T = np.matmul(X, w[:, :ndim])
    return T
