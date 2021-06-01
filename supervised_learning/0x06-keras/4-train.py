#!/usr/bin/env python3
""" Train with mini batch gradient descent """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """ method to train a model w/ mini batch GD """
    history = network.fit(data, labels, epochs=epochs, verbose=verbose,
                          batch_size=batch_size, shuffle=shuffle)
    return history
