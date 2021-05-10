#!/usr/bin/env python3
""" Deep neural network module """
import numpy as np


class DeepNeuralNetwork():
    """ DNN class """
    def __init__(self, nx, layers):
        """init the class"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or not layers:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError('layers must be a list of positive integers')

            wi = 'W'+str(i + 1)
            bi = 'b'+str(i + 1)
            if i == 0:
                self.__weights[wi] = np.random.randn(layers[i], nx)\
                                   * np.sqrt(2./nx)
            else:
                self.__weights[wi] = np.random.randn(layers[i], layers[i-1])\
                                   * np.sqrt(2/layers[i-1])
            self.__weights[bi] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ layers getter """
        return self.__L

    @property
    def cache(self):
        """ cache getter """
        return self.__cache

    @property
    def weights(self):
        """ weight getter """
        return self.__weights

    def forward_prop(self, X):
        """ DNN forward propagation method """
        self.__cache["A0"] = X
        for i in range(self.__L):
            n = str(i + 1)
            Z = np.matmul(
                self.__weights["W" + n],
                self.__cache["A" + str(i)]) + self.__weights["b" + n]
            self.__cache["A" + n] = 1/(1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache
