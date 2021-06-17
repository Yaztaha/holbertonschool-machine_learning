#!/usr/bin/env python3
""" Projection block """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ projection block kfunction """
    F11, F3, F12 = filters
    init = K.initializers.he_normal()

    conv_1a = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=s,
        kernel_initializer=init,
        padding='same'
    )(A_prev)
    norm_1a = K.layers.BatchNormalization()(conv_1a)
    relu_1a = K.layers.Activation('relu')(norm_1a)

    conv_3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        strides=1,
        kernel_initializer=init,
        padding='same'
    )(relu_1a)
    norm_3 = K.layers.BatchNormalization()(conv_3)
    relu_3 = K.layers.Activation('relu')(norm_3)

    conv_1b = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=1,
        kernel_initializer=init,
        padding='same'
    )(relu_3)
    norm_1b = K.layers.BatchNormalization()(conv_1b)

    conv_s = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=s,
        kernel_initializer=init,
        padding='same'
    )(A_prev)
    norm_s = K.layers.BatchNormalization()(conv_s)

    added = K.layers.Add()([norm_1b, norm_s])
    return K.layers.Activation('relu')(added)
