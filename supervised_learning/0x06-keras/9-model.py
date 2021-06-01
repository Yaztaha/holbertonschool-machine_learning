#!/usr/bin/env python3
""" Saves & load model """
import tensorflow.keras as K


def save_model(network, filename):
    """ method to save model  """
    network.save(filename)
    return None


def load_model(filename):
    """ method to load model """
    return K.models.load_model(filename)
