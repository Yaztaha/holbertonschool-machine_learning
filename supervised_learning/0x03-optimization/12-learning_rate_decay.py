#!/usr/bin/env python3
""" Learning rate decay """
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ learning rate decay function """
    return (tf.train.inverse_time_decay(alpha, global_step,
            decay_step, decay_rate, staircase=True))
