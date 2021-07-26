#!/usr/bin/env python3
""" Multinormal class """
import numpy as np


class MultiNormal:
    """ multinormal class  """
    def __init__(self, data):
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, 1)
        self.mean = np.expand_dims(mean, 1)
        data -= self.mean
        self.cov = np.matmul(data, data.T)/(n - 1)

    def pdf(self, x):
        """ pdf function """
        d = self.cov.shape[0]
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if (len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        x_hat = x - self.mean
        pdf = (1 / (np.sqrt((2 * np.pi)**d * np.linalg.det(self.cov)))
               * np.exp(-(np.linalg.solve(self.cov, x_hat).T.dot(x_hat)) / 2))

        return float(pdf)
