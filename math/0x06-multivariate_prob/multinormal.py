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
        if (len(x.shape) != 2):
            raise ValueError("x mush have the shape ({}, 1)".
                             format(self.cov.shape[0]))
        if (x.shape[1] != 1 or x.shape[0] != self.cov.shape[0]):
            raise ValueError("x mush have the shape ({}, 1)".
                             format(self.cov.shape[0]))
            m = x - self.mean
            det = np.linalg.det(self.cov)
            pdf_det = 1. / (np.sqrt((2 * pn.pi)**self.cov.shape[0] * det))
            pdf_inv = (np.exp(-np.linalg.solve(self.cov, m).T.dot(m)) / 2)

            pdf = pdf_det * pdf_inv
            return pdf
