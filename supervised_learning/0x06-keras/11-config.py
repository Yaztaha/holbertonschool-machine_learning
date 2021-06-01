#!/usr/bin/env python3
""" Saves & load model in json format """
import tensorflow.keras as K


def save_config(network, filename):
    """ method to save model in JSON """
    with open(filename, "w") as fd:
        fd.write(network.to_json())
    return None


def load_config(filename):
    """ method to load model JSON """
    with open(filename, "r") as fd:
        load = fd.read()
    return K.models.model_from_json(load)
