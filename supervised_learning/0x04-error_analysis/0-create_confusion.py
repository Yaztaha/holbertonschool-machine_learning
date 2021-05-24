#!/usr/bin/env python3
""" Confusion matrix """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ function that creates confusion matrix """
    return np.matmul(labels.T, logits)
