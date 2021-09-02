#!/usr/bin/env python3
""" RNN module """
import numpy as np


class RNNCell:
    """ RNN class """
    def __init__(self, i, h, o):
        """ class constructor  """
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ forward method """
        m, i = x_t.shape
        _, h = h_prev.shape
        x_ht = np.hstack((h_prev, x_t))
        h_next = np.tanh(np.matmul(x_ht, self.Wh) + self.bh)
        y_n = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_n)
        return (h_next, y)

    def softmax(self, X):
        """ softmax method """
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X), axis=-1, keepdims=True)
        return expo/expo_sum
