#!/usr/bin/env python3
""" Deep neural network module """
import numpy as np


class DeepNeuralNetwork():
    """ DNN class """
    def __init__(self, nx, layers):
        """ constructor method """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 0:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if type(layers[i]) != int or layers[i] <= 0:
                raise TypeError('layers must be a list of positive integers')

            wi = 'W'+str(i + 1)
            bi = 'b'+str(i + 1)
            if i == 0:
                self.weights[wi] = np.random.randn(layers[i], nx)\
                                  * np.sqrt(2./nx)
            else:
                self.weights[wi] = np.random.randn(layers[i], layers[i - 1])\
                                   * np.sqrt(2/layers[i - 1])
            self.weights[bi] = np.zeros((layers[i], 1))
