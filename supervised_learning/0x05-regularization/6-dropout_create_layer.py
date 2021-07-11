#!/usr/bin/env python3
""" Dropout layer creation """

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ dropout create layer function """
    dropout = tf.layers.Dropout(keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            kernel_regularizer=dropout)
    return layer(prev)
