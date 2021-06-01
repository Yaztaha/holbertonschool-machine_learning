#!/usr/bin/env python3
""" One hot encoding """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ method that make a one hot matrix """
    cat = K.utils.to_categorical(labels, num_classes=classes)
    return cat
