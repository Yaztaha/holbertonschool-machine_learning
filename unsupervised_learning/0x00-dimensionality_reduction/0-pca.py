#!/usr/bin/env python3
""" Principal component analysis """
import numpy as np


def pca(X, var=0.95):
    """Function PCA"""
    n, m = X.shape
    U, S, Vh = np.linalg.svd(X)
    sum_vars = np.sum(S)
    var_ret = S/sum_vars
    sum = 0
    count = 0
    for i in var_ret:
        sum += i
        count += 1
        if sum > var:
            break
    new_Vh = Vh.T[:, :count]
    return (new_Vh)
