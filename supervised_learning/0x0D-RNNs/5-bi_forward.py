#!/usr/bin/env python3
""" Bidirectional RNN module """
import numpy as np


class BidirectionalCell:
    """ Bidirectional class """
    def __init__(self, i, h, o):
        """ class constructor """
        self.Whf = np.random.randn(h + i, h)
        self.Whb = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h + h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ forward prop method """
        m, i = x_t.shape
        _, h = h_prev.shape
        x_ht = np.hstack((h_prev, x_t))
        h_next = np.tanh(np.matmul(x_ht, self.Whf) + self.bhf)
        return (h_next)
