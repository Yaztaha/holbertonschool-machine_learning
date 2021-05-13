#!/usr/bin/env python3
""" TensorFlow gradient decent """
import tensorflow as tf


def create_train_op(loss, alpha):
    """ gradient decent function  """
    GD_opt = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    return GD_opt
