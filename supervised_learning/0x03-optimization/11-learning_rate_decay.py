#!/usr/bin/env python3
""" Learning rate decay """
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ function of learning rate decay """
    up_alpha = alpha / (1 + decay_rate * int(global_step / decay_step))
    return (up_alpha)
