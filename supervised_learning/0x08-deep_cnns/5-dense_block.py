#!/usr/bin/env python3
""" Dense Block """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ dense block function """
    init = K.initializers.he_normal()
    output = X

    for layer in range(layers):
        norm_1 = K.layers.BatchNormalization()(output)
        relu_1 = K.layers.Activation('relu')(norm_1)
        bottleneck = K.layers.Conv2D(
            filters=growth_rate * 4,
            kernel_size=1,
            strides=1,
            kernel_initializer=init,
            padding='same'
        )(relu_1)
        norm_2 = K.layers.BatchNormalization()(bottleneck)
        relu_2 = K.layers.Activation('relu')(norm_2)
        conv_2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            strides=1,
            kernel_initializer=init,
            padding='same'
        )(relu_2)
        output = K.layers.Concatenate()([output, conv_2])

    return output, int(nb_filters + growth_rate * layers)
