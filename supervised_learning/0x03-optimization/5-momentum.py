#!/usr/bin/env python3
""" Gradient descent with momentum """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ GD with momentum function """
    v = beta1 * v + (1 - beta1) * grad
    var -= alpha * v
    return var, v
