#!/usr/bin/env python3
""" Conv same image """

import numpy as np


def convolve_grayscale_same(images, kernel):
    """ conv same img function  """
    (m, h, w), (hk, wk) = images.shape, kernel.shape

    hp = hk // 2
    wp = wk // 2
    images = np.pad(images, pad_width=((0,), (hp,), (wp,)), mode='constant')
    conv = np.zeros(shape=(m, h, w))
    for row in range(h):
        for col in range(w):
            rows = slice(row, row + hk)
            cols = slice(col, col + wk)
            part = images[:, rows, cols] * kernel
            conv[:, row, col] = np.sum(part, axis=(1, 2))
    return conv
