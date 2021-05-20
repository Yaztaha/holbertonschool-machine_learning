#!/usr/bin/env python3
""" RMSProp """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ Function RMSProp """
    up_s = (beta2 * s) + ((1 - beta2) * grad * grad)
    up_var = var - ((alpha * grad) / (((new_s)**0.5) + epsilon))
    return (up_var, up_s)
