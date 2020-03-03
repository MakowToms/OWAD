import numpy as np


def sigmoid(x, gradient=False):
    if gradient:
        return np.e**x / ((1 + np.e**x) ** 2)
    return np.e**x / (1 + np.e**x)


def linear(x, gradient=False):
    if gradient:
        return x/x
    return x


def square(x, gradient=False):
    if gradient:
        return x
    return x**2

