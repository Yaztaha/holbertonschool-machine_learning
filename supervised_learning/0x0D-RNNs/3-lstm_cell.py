#!/usr/bin/env python3
""" LSTM module """
import numpy as np


class LSTMCell:
    """ LSTM class """
    def __init__(self, i, h, o):
        """ class constructor """
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """ forward prop method """
        m, i = x_t.shape
        _, h = h_prev.shape
        st_ct_1 = np.hstack((h_prev, x_t))
        g_u = self.sigmoid(np.matmul(st_ct_1, self.Wu) + self.bu)
        g_f = self.sigmoid(np.matmul(st_ct_1, self.Wf) + self.bf)
        g_o = self.sigmoid(np.matmul(st_ct_1, self.Wo) + self.bo)
        c_tilde = np.tanh(np.matmul(st_ct_1, self.Wc) + self.bc)
        c_next = (g_u * c_tilde) + (g_f * c_prev)
        h_next = g_o * np.tanh(c_next)
        y_n = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_n)
        return (h_next, c_next, y)

    def softmax(self, X):
        """ softmax method """
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X), axis=-1, keepdims=True)
        return expo/expo_sum

    def sigmoid(self, X):
        """ sigmoid method """
        return 1/(1 + np.exp(-X))
