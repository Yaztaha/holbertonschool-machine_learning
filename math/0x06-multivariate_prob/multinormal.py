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
        if not(isinstance(x, np.ndarray)):
            raise TypeError("x must be a numpy.ndarray")
        cov = self.cov
        if (len(x.shape) is not 2 or x.shape[1] is not 1
                or x.shape[0] is not cov.shape[0]):
            raise ValueError("x must have the shape ({d}, 1)".
                             format(cov.shape[0]))
        else:
            cov = self.cov
            inv_cov = np.linalg.inv(cov)
            mean = self.mean
            D = cov.shape[0]
            det_cov = np.linalg.det(cov)
            den = np.sqrt(np.power((2 * np.pi), D) * det_cov)
            y = np.matmul((x - mean).T, inv_cov)
            pdf = (1 / den) * np.exp(-1 * np.matmul(y, (x - mean)) / 2)
            return pdf.reshape(-1)[0]
