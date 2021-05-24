#!/usr/bin/env python3
""" F1_score of confusion matrix """


import numpy as np
sntv = __import__('1-sensitivity').sensitivity
prcs = __import__('2-precision').precision


def f1_score(confusion):
    """ function that return f1 score """
    _sntv = sensitivity(confusion)
    _prcs = precision(confusion)
    F1_score = 2 * ((_prcs * _sntv) / (_prcs + _sntv))

    return F1_score
