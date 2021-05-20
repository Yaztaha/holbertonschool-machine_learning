#!/usr/bin/env python3
""" Adam optimization """
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ function of Adam """
    train = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
    return(train)
