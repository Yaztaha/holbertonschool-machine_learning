#!/usr/bin/env python3
""" Integration """


def poly_integral(poly, C=0):
    """method of new coeffiecients for an integral"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None
    if poly == [0]:
        return [C]

    lis = [C]
    for i in range(len(poly)):
        if (poly[i] % (i + 1)) == 0:
            new = int(poly[i]/(i + 1))
        else:
            new = poly[i]/(i + 1)
            lis.append(new)
    return lis
