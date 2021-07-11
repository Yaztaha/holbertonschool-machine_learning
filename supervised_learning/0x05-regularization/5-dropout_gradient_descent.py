#!/usr/bin/env python3
""" Gradient descent with dropout """

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ drouptout gd funcion """
    m = Y.shape[1]
    for i in reversed(range(L)):
        key_w = 'W' + str(i + 1)
        key_b = 'b' + str(i + 1)
        key_cache = 'A' + str(i + 1)
        key_cache_dw = 'A' + str(i)
        A = cache[key_cache]
        A_dw = cache[key_cache_dw]
        if i == L - 1:
            dz = A - Y
            W = weights[key_w]
        else:
            da = 1 - (A * A)
            dz = np.matmul(W.T, dz)
            dz = dz * da * cache["D{}".format(i + 1)]
            dz = dz / keep_prob
            W = weights[key_w]
        dw = np.matmul(A_dw, dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights[key_w] = weights[key_w] - alpha * dw.T
        weights[key_b] = weights[key_b] - alpha * db
