#!/usr/bin/env python3
""" Neuron module """

import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ gradient descent method """
        dZ = A - Y
        dW = (dZ @ X.T) / X.shape[1]
        db = np.sum(dZ) / X.shape[1]

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ method to train neuron """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        xaxis = []
        yaxis = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            cost = self.cost(Y, self.__A)

            if verbose is True:
                if i % step == 0:
                    print("Cost after {} iterations: {}".format(i, cost))
                    xaxis.append(i)
                    yaxis.append(cost)
                
        if graph is True:
            plt.plot(xaxis, yaxis, 'tab:blue', '-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
