#!/usr/bin/env python3
""" Deep neural network module """
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def cost(self, Y, A):
        """ DNN cost method """
        cost = -(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A))).mean()
        return cost

    def evaluate(self, X, Y):
        """ DNN evaluation method """
        A = self.forward_prop(X)[0]
        cost = self.cost(Y, A)
        A = np.where(A > 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ DNN gradient decent method """
        m = Y.shape[1]
        dz = self.__cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A = "A{}".format(i-1)
            W = "W{}".format(i)
            b = "b{}".format(i)

            dw = np.matmul(dz, self.__cache[A].T) * (1/m)
            db = np.sum(dz, axis=1, keepdims=True) * (1/m)
            dz = np.matmul(self.__weights[W].T, dz) * (self.__cache[A] *
                                                       (1 - self.__cache[A]))

            self.__weights[W] -= alpha * dw
            self.__weights[b] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ optimized DNN training method """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        xaxis = []
        yaxis = []

        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            cost = self.cost(Y, self.__cache["A{}".format(self.__L)])

            if verbose:
                if i % step == 0:
                    xaxis.append(i)
                    yaxis.append(cost)
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(xaxis, yaxis, 'tab:blue', '-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Trainig Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """ method to save into pickle file """
        if filename[-4:] is not '.plk':
            filename = filename + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """ load pickle file method """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            return None
