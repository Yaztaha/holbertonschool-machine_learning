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

    def cost(self, Y, A):
        """ cost method """
        loss = -1 * (Y * np.log(A) + (1 - Y) *
                     np.log(1.0000001 - A))
        cost = np.sum(loss) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """ evaluate method """
        self.forward_prop(X)
        evaluation = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return evaluation, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ gradient descent method """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.matmul(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 -= dW1 * alpha
        self.__b1 -= db1 * alpha
        self.__W2 -= dW2 * alpha
        self.__b2 -= db2 * alpha

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ neuron training method """
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
