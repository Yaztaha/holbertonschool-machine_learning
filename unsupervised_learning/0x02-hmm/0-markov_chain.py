#!/usr/bin/env python3
""" Markov chain """
import numpy as np


def markov_chain(P, s, t=1):
    """ markov chain function """
    try:
        if (not isinstance(P, np.ndarray) or
                not isinstance(s, np.ndarray)):
            return None
        if P.shape[0] is not P.shape[1]:
            return None
        if len(P.shape) is not 2:
            return None
        a = np.sum(P) / P.shape[0]
        if a != 1:
            return None
        if type(t) is not int or t <= 1:
            return None
        return np.matmul(s, np.linalg.matrix_power(P, t))
    except Exception:
        return None
