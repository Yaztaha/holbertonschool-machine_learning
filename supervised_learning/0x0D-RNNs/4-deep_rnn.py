#!/usr/bin/env python3
""" Deep RNN """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ deep RNN function """
    t, m, i = X.shape
    l, _, h = h_0.shape
    H = np.zeros((t+1, l, m, h))
    H[0] = h_0
    for iter in range(1, t + 1):
        h_prev = h_0
        for layers in range(l):
            if layers == 0:
                h_next, y = rnn_cells[layers].forward(H[iter-1,
                                                        layers], X[iter-1])
            else:
                h_next, y = rnn_cells[layers].forward(H[iter-1,
                                                        layers], h_prev)
            h_prev = h_next
            H[iter, layers] = h_next
        if iter-1 == 0:
            yv = y.shape[1]
            y_r = np.reshape(y, (1, m, yv))
            Y = y_r
        else:
            y_r = np.reshape(y, (1, m, yv))
            Y = np.vstack((Y, y_r))
    return(H, Y)
