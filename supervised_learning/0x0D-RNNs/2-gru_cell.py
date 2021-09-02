#!/usr/bin/env python3
""" GRU cell module """
import numpy as np


class GRUCell:
    """ GRU cell class """
    def __init__(self, i, h, o):
        """ class constructor """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ forward prop method """
        m, i = x_t.shape
        _, h = h_prev.shape
        st_ct_1 = np.hstack((h_prev, x_t))
        g_u = self.sigmoid(np.matmul(st_ct_1, self.Wz) + self.bz)
        g_r = self.sigmoid(np.matmul(st_ct_1, self.Wr) + self.br)
        st_ct_full = np.hstack(((g_r * h_prev), x_t))
        c_tilde = np.tanh(np.matmul(st_ct_full, self.Wh) + self.bh)
        h_next = (g_u * c_tilde) + ((1-g_u) * h_prev)
        y_n = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_n)
        return (h_next, y)

    def softmax(self, X):
        """ softmax method """
        np.exp(X)
        expo_sum = np.sum(np.exp(X), axis=-1, keepdims=True)
        return expo/expo_sum

    def sigmoid(self, X):
        """ sigmoid method """
        return 1/(1 + np.exp(-X))
