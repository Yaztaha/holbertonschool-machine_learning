#!/usr/bin/env python3
""" Bidirctional RNN forward prop """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ bi_rnn forward prop """
    t, m, i = X.shape
    h = h_0.shape[1]
    h_for = np.zeros((t, m, h))
    h_back = np.zeros((t, m, h))
    h_ft = h_0
    h_bt = h_t
    for i in range(t):
        x_ft = X[i]
        x_bt = X[-(i+1)]
        h_ft = bi_cell.forward(h_ft, x_ft)
        h_bt = bi_cell.backward(h_bt, x_bt)
        h_for[i] = h_ft
        h_back[-(i+1)] = h_bt
    H = np.concatenate((h_for, h_back), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
