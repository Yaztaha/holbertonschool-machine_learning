#!/usr/bin/env python3
"""Shape of a matrix"""

def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""
    if not type(matrix) == list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
