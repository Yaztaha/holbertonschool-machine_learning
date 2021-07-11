#!/usr/bin/env python3
""" L2 regularization with tensorflow """

import tensorflow as tf


def l2_reg_cost(cost):
    """ l2 reg with TF function  """
    return cost + tf.losses.get_regularization_losses()
