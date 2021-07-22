#!/usr/bin/env python3
""" Matrix determinant """


def determinant(matrix):
    """ matrix determinant function """
    if matrix == [[]]:
        return 1
    if (type(matrix) is not list or matrix == [] or
       any([type(el) != list for el in matrix])):
        raise TypeError('matrix must be a list of lists')
    if len(matrix) != len(matrix[0]):
        raise ValueError('matrix must be a square matrix')
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i, n in enumerate(matrix[0]):
        m = matrix[1:]
        filt = [[num for idx, num in enumerate(col) if idx != i] for col in m]
        det += (n * (-1) ** i * determinant(filt))
    return det
