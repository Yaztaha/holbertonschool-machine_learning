#!/usr/bin/env python3
""" Specifity of confusion matrix """
import numpy as np


def specificity(confusion):
    """ function that return specifity of a cm """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (FP + FN + TP)
    SPCF = TN / (FP + TN)

    return SPCF
