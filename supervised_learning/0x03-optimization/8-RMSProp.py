#!/usr/bin/env python3
""" RMSProp """
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ RMSProp function """
    train = tf.train.RMSPropOptimizer(alpha, decay=beta2,
                                      epsilon=epsilon).minimize(loss)
    return(train)
