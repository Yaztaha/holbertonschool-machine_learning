#!/usr/bin/env python3
"""flip a 2d matrix"""


def matrix_transpose(matrix):
    """transpose of a 2d matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
