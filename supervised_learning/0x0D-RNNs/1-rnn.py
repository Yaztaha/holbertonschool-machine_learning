#!/usr/bin/env python3
""" RNN """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ RNN forward prop function """
    t, m, i = X.shape
    _, h = h_0.shape
    H = h_0
    H = np.reshape(H, (1, m, h))

    for iter in range(t):
        h_next, y = rnn_cell.forward(h_0, X[iter, :, :])
        h_next_r = np.reshape(h_next, (1, m, h))
        H = np.vstack((H, h_next_r))
        if iter == 0:
            yv = y.shape[1]
            y_r = np.reshape(y, (1, m, yv))
            Y = y_r
        else:
            y_r = np.reshape(y, (1, m, yv))
            Y = np.vstack((Y, y_r))
        h_0 = h_next
    return(H, Y)
