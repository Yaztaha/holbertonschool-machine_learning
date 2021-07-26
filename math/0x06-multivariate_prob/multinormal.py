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
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]

        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        det = np.linalg.det(self.cov)
        b = (((2 * np.pi)**d) * det)**(.5)
        m = (x - self.mean)
        i = np.linalg.inv(self.cov)
        return ((1/b) * np.exp((-1/2) * m.T @ i @ m))[0][0]
