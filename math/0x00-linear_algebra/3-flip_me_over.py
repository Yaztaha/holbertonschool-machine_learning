#!/usr/bin/env python3
"""Transpose of 2D matrix"""


def matrix_transpose(matrix):
    """Transpose of a 2D matrix"""
    return [[matrix[t][a] for t in range(len(matrix))]
            for t in range(len(matrix[0]))]
