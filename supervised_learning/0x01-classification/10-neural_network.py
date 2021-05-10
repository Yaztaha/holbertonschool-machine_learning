#!/usr/bin/env python3
""" Neural network module """
import numpy as np


class NeuralNetwork:
    """ neural network class """
    def __init__(self, nx, nodes):
        """ constructor method """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) != int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def sigmoid(self, x):
        """ sigmoid method """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def forward_prop(self, X):
        """ forward propagation method """
        z1 = np.matmul(self.__W1, X) + self.__b1
        A1 = self.sigmoid(z1)
        self.__A1 = A1
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        A2 = self.sigmoid(z2)
        self.__A2 = A2
        return self.__A1, self.__A2
