#!/usr/bin/env python3
""" TensorFlow layer creation """
import tensorflow as tf


def create_layer(prev, n, activation):
    """ layer creation function """
    w_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    create_layers = tf.layers.Dense(units=n, activation=activation,
                                    kernel_initializer=w_init,
                                    name='layer')
    return create_layers(prev)
