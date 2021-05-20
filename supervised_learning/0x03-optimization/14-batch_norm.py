#!/usr/bin/env python3
""" Batch norm """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ batch norm function """
    initial = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    my_layer = tf.layers.Dense(units=n, activation=None,
                               kernel_initializer=initial)
    mean, variance = tf.nn.moments(my_layer(prev), axes=[0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]),
                        name='gamma', trainable=True)
    BN2 = tf.nn.batch_normalization(my_layer(prev), mean=mean,
                                    variance=variance,
                                    variance_epsilon=1e-8,
                                    offset=beta, scale=gamma)
    return(activation(BN2))
