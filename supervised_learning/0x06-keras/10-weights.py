#!/usr/bin/env python3
""" Saves & load model's weights """
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ method to save weights """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """ method to load weights """
    network.load_weights(filename)
    return None
