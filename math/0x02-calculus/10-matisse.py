#!/usr/bin/env python3
""" Derivative of polynomial """


def poly_derivative(poly):
    """ Derivative of polynomial function """

    if poly is None or len(poly) == 0:
        return None

    if len(poly) == 1:
        return None

    if type(poly) is not list:
        return None

    for element in poly:
        if not isinstance(element, (int, float)):
            return None

    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])
    return derivative
