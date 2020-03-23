import numpy as np


class Layer:
    def __init__(self, inp, out, act, initialize_weights):
        self.n_input = inp
        self.n_output = out
        self.activation = act
        self.initialization_weight_and_bias(inp, out, initialize_weights)

    def initialization_weight_and_bias(self, inp, out, initialize_weights):
        self.momentum_weights = np.zeros([out, inp])
        self.momentum_bias = np.zeros([1, out]).transpose()
        self.eg_weights = np.zeros([out, inp])
        self.eg_bias = np.zeros([1, out]).transpose()
        if initialize_weights == 'zeros':
            self.weights = np.zeros([out, inp])
            self.bias = np.zeros([1, out]).transpose()
        elif initialize_weights == 'norm':
            self.weights_from_normal(inp, out)
        elif initialize_weights == 'Xavier':
            self.weights_from_normal(inp, out)
            self.weights = self.weights * np.sqrt(6) / np.sqrt(self.n_input + self.n_output)
            self.bias = self.bias * np.sqrt(6) / np.sqrt(self.n_input + self.n_output)

    def weights_from_normal(self, inp, out):
        self.weights = np.random.rand(out, inp) - (np.ones([out, inp]) * 1 / 2)
        self.bias = np.random.rand(1, out).transpose() - (np.ones([1, out]).transpose() * 1 / 2)

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

    def update_weights_and_bias_backward(self, previous_result, eta, lambda_momentum, beta, moment_type):
        learning_rate = - eta
        delta_weights = self.backward_error @ previous_result.transpose()
        delta_bias = np.sum(self.backward_error)
        if moment_type == 'normal':
            self.weights = self.weights + learning_rate * delta_weights
            self.bias = self.bias + learning_rate * delta_bias
        elif moment_type == 'momentum':
            self.momentum_weights = delta_weights + self.momentum_weights * lambda_momentum
            self.momentum_bias = delta_bias + self.momentum_bias * lambda_momentum
            self.weights = self.weights + learning_rate * self.momentum_weights
            self.bias = self.bias + learning_rate * self.momentum_bias
        elif moment_type == 'RMSProp':
            eps = 10 ** (-9)
            self.eg_weights = (1-beta) * (delta_weights * delta_weights) + self.eg_weights * beta
            self.eg_bias = (1-beta) * (delta_bias * delta_bias) + self.eg_bias * beta
            self.weights = self.weights + learning_rate * delta_weights / (np.sqrt(self.eg_weights) + np.ones(self.eg_weights.shape) * eps)
            self.bias = self.bias + learning_rate * delta_bias / (np.sqrt(self.eg_bias) + np.ones(self.eg_bias.shape) * eps)
