#!/usr/bin/env python3
""" Transition layer """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ transition layer function """
    norm = K.layers.BatchNormalization()(X)
    relu = K.layers.Activation('relu')(norm)
    conv = K.layers.Conv2D(
        filters=int(nb_filters * compression),
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=K.initializers.he_normal()
    )(relu)
    pool = K.layers.AveragePooling2D()(conv)
    return pool, int(nb_filters * compression)
