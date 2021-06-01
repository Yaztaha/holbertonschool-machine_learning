#!/usr/bin/env python3
""" Model prediction """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ method to make prediction """
    return network.predict(x=data, verbose=verbose)
