#!/usr/bin/env python3
""" Adam optimization """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ Adam optimization function """
    Vd = (beta1 * v) + ((1 - beta1) * grad)
    Sd = (beta2 * s) + ((1 - beta2) * grad * grad)

    up_prom_corr = Vd / (1 - beta1 ** t)
    up_s_corr = Sd / (1 - beta2 ** t)

    w = var - alpha * (up_prom_corr / ((up_s_corr ** (0.5)) + epsilon))
    return (w, Vd, Sd)
