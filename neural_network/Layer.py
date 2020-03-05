import numpy as np


class Layer:
    def __init__(self, inp, out, act, eta):
        self.n_input = inp
        self.n_output = out
        self.activation = act
        self.eta = eta
        self.weights = np.ones([out, inp])
        self.bias = np.ones([1, out]).transpose()

    def forward(self, x):
        self.forward_without_activation = np.matmul(self.weights, x) + self.bias
        self.forward_with_activation = self.activation(self.forward_without_activation)
        return self.forward_with_activation

    def backward_last_error(self, true, predict):
        self.__forward_gradient__()
        self.backward_error = (predict.transpose()-true.transpose()) * self.forward_gradient

    def backward_other_error(self, ekWk):
        self.__forward_gradient__()
        self.backward_error = ekWk.transpose() * self.forward_gradient

    def __forward_gradient__(self):
        self.forward_gradient = self.activation(self.forward_without_activation, gradient=True)

    def ekWk(self):
        return self.backward_error.transpose() @ self.weights

    def update_weights_and_bias_backward(self, previous_result):
        delta_weights = - self.eta / previous_result.shape[1] * self.backward_error @ previous_result.transpose()
        delta_bias = - self.eta / previous_result.shape[1] * np.sum(self.backward_error)
        self.weights = self.weights + delta_weights
        self.bias = self.bias + delta_bias
