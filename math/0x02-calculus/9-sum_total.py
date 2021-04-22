#!/usr/bin/env python3
""" Sigma notation & Faulhaber's formula """


def summation_i_squared(n):
    """ Faulhaber formula for summation of i^2 """
    if type(n) is not int or n < 1:
        return None
    sigma_sum = (n * (n + 1) * ((2 * n) + 1)) / 6
    return int(sigma_sum)
