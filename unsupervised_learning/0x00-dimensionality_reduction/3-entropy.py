#!/usr/bin/env python3
""" Shannon entropy and P affinities """
import numpy as np


def HP(Di, beta):
    """ Shannon entropy and P affinities function """
    Pi = np.exp(-Di * beta)
    Pi = Pi / np.sum(Pi)
    Hi = np.sum(-Pi * np.log2(Pi))
    return(Hi, Pi)
