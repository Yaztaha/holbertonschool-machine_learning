#!/usr/bin/env python3
""" TensorFlow softmax entropy """
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ softmax entropy loss function """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    return loss
