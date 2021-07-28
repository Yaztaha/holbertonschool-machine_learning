#!/usr/bin/env python3
""" Principal component analysis """
import numpy as np


def pca(X, ndim):
    """Function PCA2 with dim"""
    m, n = X.shape
    X = np.mean(X, axis=0) - X
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=True)
    new_Vh = Vh.T[:, :ndim]
    T = np.matmul(X, new_Vh)
    return (T)
