#!/usr/bin/env python3
""" Convolution on grayscale img """

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ method that convolve on grayscale img """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    n_h = h - kh + 1
    n_w = w - kw + 1
    convolved = np.zeros([m, n_h, n_w])
    for x in range(n_h):
        for y in range(n_w):
            image = images[:, x:x+kh, y:y+kw]
            convolved[:, x, y] = np.multiply(image, kernel).sum(axis=(1, 2))
    return convolved
