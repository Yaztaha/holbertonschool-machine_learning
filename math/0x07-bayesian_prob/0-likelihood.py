#!/usr/bin/env python3
""")Likelihood module """

import numpy as np


def likelihood(x, n, P):
    """ likelihood function """
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) is not 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    factor2 = (P ** x) * (1 - P) ** (n - x)
    factor1 = np.math.factorial(n) / \
        (np.math.factorial(x) * np.math.factorial(n - x))
    return factor1 * factor2
