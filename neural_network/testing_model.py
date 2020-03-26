import numpy as np


def MSE(x, y):
    return np.mean((x - y) ** 2)


def accuracy(result, y_numeric):
    return np.sum(y_numeric.transpose()==result)/y_numeric.shape[0]
