#!/usr/bin/env python3
""" Integration """


def poly_integral(poly, C=0):
    """method of new coeffiecients for an integral"""
    if poly == [] or type(poly) is not list or type(C) is not int:
        return None
    if len(poly) == 1:
        return [C]
    result = [C]

    for i in range(len(poly)):
        integ = poly[i] / (i+1)
        if inte.is_integer():
            integ = int(r)
            result.append(integ)

        return result
