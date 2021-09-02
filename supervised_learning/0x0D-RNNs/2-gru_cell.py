#!/usr/bin/env python3
""" GRU module """
import numpy as np


class GRUCell:
    """ GRU class """
    def __init__(self, i, h, o):
        """ class constructor """
        self.Wz = np.random.normal(size=(i+h, h))
        self.Wr = np.random.normal(size=(i+h, h))
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ forwad prop methjod """
        new_w = np.concatenate((h_prev, x_t), axis=1)
        z = np.matmul(new_w, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z))
        r = np.matmul(new_w, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r))

        new_w2 = np.concatenate((r * h_prev, x_t), axis=1)
        h = np.tanh(np.matmul(new_w2, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h

        y_mul = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_mul) / np.sum(np.exp(y_mul), axis=1, keepdims=True)
        return h_next, y
