import numpy as np


def sigmoid(x, gradient=False):
    if gradient:
        return np.e**x / ((1 + np.e**x) ** 2)
    return np.e**x / (1 + np.e**x)


def linear(x, gradient=False):
    if gradient:
        return np.ones(x.shape)
    return x


def square(x, gradient=False):
    if gradient:
        return x
    return x**2


def softmax(x, gradient=False):
    if gradient:
        # in fact it not returns all gradient
        # since in numpy it is not possible to multiply
        # tensor 3d by tensor 2d in a way that is needed
        #
        # it returns base to compute all gradient
        return - np.e**x / (np.sum(np.e**x, axis=0) ** 2)
    return np.e**x / np.sum(np.e**x, axis=0)
