#!/usr/bin/env python3
""" Test a model """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ method to test a model """
    return network.evaluate(x=data, y=labels, verbose=verbose)
