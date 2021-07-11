#!/usr/bin/env python3
""" L2 regularization """

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ l2 regularization function  """
    L2 = 0
    for i in range(L):
        w = weights["W{}".format(i + 1)]
        L2 += 1/m * lambtha/2 * np.linalg.norm(w)
    return L2 + cost
