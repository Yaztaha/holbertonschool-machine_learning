#!/usr/bin/env python3
""" Symmetric P affinities """
import numpy as np


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """ symmetric p affinities function """
    n = X.shape[0]

    D, P, beta, H = P_init(X, perplexity)

    for i in range(n):
        b_min = None
        b_max = None

        Di = np.delete(D[i], i, axis=0)

        Hi, Pi = HP(Di, beta[i])

        Hdiff = Hi - H
        tries = 0
        while np.abs(Hdiff) > tol and tries < 1000:
            if Hdiff > 0:
                b_min = beta[i, 0]
                if b_max is None:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + b_max) / 2
            else:
                b_max = beta[i, 0]
                if b_min is None:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + b_min) / 2
            Hi, Pi = HP(Di, beta[i])
            Hdiff = Hi - H
        Pi = np.insert(Pi, i, 0)
        P[i] = Pi

    P = (P.T + P) / (2 * n)
    return P
