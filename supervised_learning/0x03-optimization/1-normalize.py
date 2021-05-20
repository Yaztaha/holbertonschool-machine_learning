#!/usr/bin/env python3
""" Normalize matrix """


def normalize(X, m, s):
    """ matrix norm function """
    mtrxnorm = (X - m) / s
    return mtrxnorm
