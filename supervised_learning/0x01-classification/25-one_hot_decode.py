#!/usr/bin/env python3
""" One hot decode """
import numpy as np


def one_hot_decode(one_hot):
    """ one hot decoding method """
    if type(one_hot) != np.ndarray:
        return None

    return(np.argmax(one_hot, axis=0))
