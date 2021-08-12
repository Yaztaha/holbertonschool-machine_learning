#!/usr/bin/env python3
""" Forward algo """
import numpy as np


def forward(Observations, Emission, Transition, Initial):
    """ forward algo function """
    try:
        T = Observations.shape[0]
        N = Transition.shape[0]
        alpha = np.zeros((N, T))
        test = Emission[:, Observations[0]]
        alpha[:, 0] = Initial.T * Emission[:, Observations[0]]
        for t in range(1, T):
            for n in range(N):
                a1 = alpha[:, t-1] * Transition[:, n]
                alpha[n, t] = np.sum(Transition[:, n] *
                                     alpha[:, t-1] *
                                     Emission[n, Observations[t]])
        P = np.sum(alpha[:, -1:])
        return P, alpha
    except Exception:
        return None, None
