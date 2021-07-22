#!/usr/bin/env python3
""" Definiteness of a matrix """
import numpy as np


def definiteness(matrix):
    """function that calculates the definiteness of a matrix """
    if type(matrix) != np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    if matrix.shape[0] != matrix.shape[1] or (matrix != matrix.T).any():
        return None
    try:
        eig = np.linalg.eigvals(matrix)
        if all(eig > 0):
            return 'Positive definite'
        if all(eig >= 0):
            return 'Positive semi-definite'
        if all(eig < 0):
            return 'Negative definite'
        if all(eig <= 0):
            return 'Negative semi-definite'
        return 'Indefinite'
    except Exception:
        return None
