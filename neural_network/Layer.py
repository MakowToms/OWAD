import numpy as np


class Layer:
    def __init__(self, inp, out, act):
        self.n_input = inp
        self.n_output = out
        self.activation = act
        self.weights = np.ones([out, inp])
        self.bias = np.ones([1, out]).transpose()

    def forward(self, x):
        result = np.matmul(self.weights, x) + self.bias
        return self.activation(result)

