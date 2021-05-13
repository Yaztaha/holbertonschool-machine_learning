#!/usr/bin/env python3
""" TensoFlow mean loss"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ function to calculate loss """
    check = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    loss = tf.reduce_mean(tf.cast(check, tf.float32))

    return loss
