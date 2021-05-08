#!/usr/bin/env python3
""" Neuron module """

import numpy as np


class Neuron:
    """ Neuron class """
    def __init__(self, nx):
        """ class constructor """
        self.nx = nx
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be positive')

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ weight getter """
        return self.__W

    @property
    def b(self):
        """ bias getter """
        return self.__b

    @property
    def A(self):
        """ output getter """
        return self.__A

    def forward_prop(self, X):
        """ forward propagation method """
        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1+np.exp(-x))
        return self.__A

    def cost(self, Y, A):
        """ logistic regression cost method """
        Cost_Sum = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return (1 / A.shape[1]) * Cost_Sum

    def evaluate(self, X, Y):
        """ method to evaluate neuron prediction """
        self.forward_prop(X)
        evaluation = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return evaluation, cost
