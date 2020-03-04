import numpy as np


class Layer:
    def __init__(self, inp, out, act, eta):
        self.n_input = inp
        self.n_output = out
        self.activation = act
        self.eta = eta
        self.weights = np.ones([out, inp])
        self.bias = np.ones([1, out]).transpose()
        self.forward_without_activation = None
        self.forward_with_activation = None
        self.forward_gradient = None
        self.backward_error = None
        self.delta_weights = None
        self.delta_bias = None

    def forward(self, x):
        self.forward_without_activation = np.matmul(self.weights, x) + self.bias
        self.forward_with_activation = self.activation(self.forward_without_activation)
        return self.forward_with_activation

    def backward_last_error(self, true, predict):
        print(true.shape)
        print(predict.shape)
        self.forward_gradient = self.activation(self.forward_without_activation, gradient=True)
        print(self.weights.shape)
        print(self.forward_with_activation.shape)
        print(self.forward_gradient.shape)
        self.backward_error = (predict.transpose()-true.transpose()) * self.forward_gradient
        print(self.backward_error.shape)

    def backward_other_error(self, ekWk):
        self.forward_gradient = self.activation(self.forward_without_activation, gradient=True)
        self.backward_error = ekWk.transpose() * self.forward_gradient
        print(ekWk.shape)
        print(self.weights.shape)
        print(self.forward_with_activation.shape)
        print(self.forward_gradient.shape)
        print(self.backward_error.shape)

    def ekWk(self):
        print(self.n_input)
        print(self.n_output)
        return np.matmul(self.backward_error.transpose(), self.weights)

    def update_weights_backward(self, previous_result):
        print('error', self.backward_error.shape, ' previous', previous_result.shape)
        self.delta_weights = - self.eta * np.matmul(self.backward_error, previous_result.transpose())
        print(self.delta_weights.shape)
        self.delta_bias = - self.eta * np.sum(self.backward_error)
        print(self.weights.shape)
        self.weights = self.weights + self.delta_weights
        print(self.weights.shape)
        self.bias = self.bias + self.delta_bias
