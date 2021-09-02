#!/usr/bin/env python3
""" Bidirctional RNN forward prop """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ bi_rnn forward prop """
    t, m, i = X.shape
    _, h = h_0.shape
    H = np.zeros((t, m, 2*h))
    H[0, :, 0:h] = h_0
    H[t-1, :, h:2*h] = h_t
    for iter in range(t):
        H[iter, :, 0:h] = bi_cell.forward(H[iter-1, :, 0:h], X[iter])
        H[t-iter-2, :, h:2*h] = bi_cell.backward(H[t-1-iter, :, h:2*h],
                                                 X[t-1-iter])
    H[t-1, :, h:2*h] = 0
    Y = bi_cell.output(H)
    return(H, Y)
