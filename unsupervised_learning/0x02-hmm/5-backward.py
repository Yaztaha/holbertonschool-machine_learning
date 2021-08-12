#!/usr/bin/env python3
""" Backward algo """
import numpy as np


def backward(Observations, Emission, Transition, Initial):
    """ backward algo function """
    try:
        T = Observations.shape[0]
        N = Transition.shape[0]
        beta = np.zeros((N, T))
        beta[:, T-1] = np.ones((N))
        for t in range(T - 2, -1, -1):
            for n in range(N):
                beta[n, t] = np.sum((Transition[n, :]*beta[:, t+1]) *
                                    Emission[:, Observations[t+1]])
        P = np.sum(beta[:, 0] * Emission[:, Observations[0]] * Initial[:, 0])
        return P, beta
    except Exception:
        return None, None
