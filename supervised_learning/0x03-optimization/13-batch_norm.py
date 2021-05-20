#!/usr/bin/env python3
""" Batch norm """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ batch norm function """
    Z_norm = ((Z - np.mean(Z, axis=0))
              / ((np.var(Z, axis=0) + epsilon)**0.5))
    return(gamma * Z_norm + beta)
