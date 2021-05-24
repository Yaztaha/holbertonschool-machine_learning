#!/usr/bin/env python3
""" F1_score of confusion matrix """


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ function that return f1 score """
    PPV = precision(confusion)
    TPR = sensitivity(confusion)
    F1 = 2 * PPV * TPR / (PPV + TPR)
    return F1
