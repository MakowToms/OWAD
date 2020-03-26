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
    # print(x.shape)
    if gradient:
        return - np.e**x / np.sum(np.e**x, axis=0) ** 2
        # result = np.zeros([x.shape[0], x.shape[1], x.shape[0]])
        # for i in range(x.shape[1]):
        #     result[:, i, :] = softmax_gradient_for_one_observation(x[:, i])
        # return result
    return np.e**x / np.sum(np.e**x, axis=0)


def softmax_gradient_for_one_observation(x):
    ez = np.e ** x
    sum_ez = np.sum(ez, axis=0)
    diag = np.diag(sum_ez * ez)
    result = diag - (np.ones([x.shape[0], x.shape[0]]) * ez).transpose() * ez
    return result / (sum_ez) ** 2
