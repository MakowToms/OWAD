import numpy as np


def sigmoid(x):
    return np.e**x / (1 + np.e**x)


def linear(x):
    return x


def square(x):
    return x**2

