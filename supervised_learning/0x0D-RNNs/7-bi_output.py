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

    def backward(self, h_next, x_t):
        """ backprop method """
        m, i = x_t.shape
        _, h = h_next.shape
        x_ht = np.hstack((h_next, x_t))
        h_next = np.tanh(np.matmul(x_ht, self.Whb) + self.bhb)
        return (h_next)

    def output(self, H):
        """ output method """
        Y_n = np.matmul(H, self.Wy) + self.by
        Y = self.softmax(Y_n)
        return Y

    def softmax(self, X):
        """ softmax method """
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X), axis=-1, keepdims=True)
        return expo/expo_sum

    def sigmoid(self, X):
        """ simoid function """
        return (1/(1 + np.exp(-X)))
