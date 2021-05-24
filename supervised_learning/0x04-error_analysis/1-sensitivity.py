#!/usr/bin/env python3
""" Sensitivity of confusion matrix """

import numpy as np


def sensitivity(confusion):
    """ function that return sensitivity of classes"""
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    SNTV = TP / (TP + FN)

    return SNTV
