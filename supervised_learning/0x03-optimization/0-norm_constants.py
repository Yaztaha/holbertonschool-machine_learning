#!/usr/bin/env python3
""" Normalization constants """
import numpy as np


def normalization_constants(X):
    """ mean & standard deviation function """
    m = X.shape[0]
    mean = np.sum(X, axis=0) / m
    std_dev = np.sqrt(np.sum((X - mean) ** 2, axis=0) / m)
    return mean, std_dev
