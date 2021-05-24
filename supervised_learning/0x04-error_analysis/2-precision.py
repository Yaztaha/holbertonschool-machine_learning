#!/usr/bin/env python3
""" Precision of confusion matrix """

import numpy as np


def precision(confusion):
    """ function that return presicion of confusion matrix """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    PRCS = TP / (TP + FP)

    return PRCS
