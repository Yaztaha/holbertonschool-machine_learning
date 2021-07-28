#!/usr/bin/env python3
""" Cost of t-SNE """
import numpy as np


def cost(P, Q):
    """ cost function """
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    C = np.sum(P * np.log(P / Q))
    return C
