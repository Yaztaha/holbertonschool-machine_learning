#!/usr/bin/env python3
""" Derivative of polynomial """


def poly_derivative(poly):
    """ Derivative of polynomial function """

    if not isinstance(poly, list) or len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    poly_deriv = []
    for coef in range(len(poly)):
        if coef == 0:
            continue
        poly_deriv.append(coef * poly[coef])
    return poly_deriv
