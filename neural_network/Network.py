from neural_network.Layer import Layer
import math


class Network:
    def __init__(self, inp, hidden, out, activations, eta=0.001):
        if hidden is None:
            hidden = []
        assert len(hidden) + 1 == len(activations)

        layers = []
        hidden.insert(0, inp)
        hidden.append(out)
        for i in range(len(activations)):
            layers.append(Layer(hidden[i], hidden[i+1], activations[i], eta))
        self.layers = layers
        self.n_layers = len(layers)
        self.results = []

    def forward(self, x):
        self.results = []
        x = x.transpose()
        for i in range(self.n_layers):
            x = self.layers[i].forward(x)
            self.results.append(x)
        return x.transpose()

    def backward(self, x, y, batch_size=50):
        n_batches = math.ceil(x.shape[0]/batch_size)
        for i in range(n_batches):
            self.backward_batch(x[i*batch_size:(i+1)*batch_size, :], y[i*batch_size:(i+1)*batch_size, :])

    def backward_batch(self, x, y):
        y_predict = self.forward(x)
        self.layers[self.n_layers-1].backward_last_error(y, y_predict)
        for i in range(self.n_layers-2, -1, -1):
            ekWk = self.layers[i+1].ekWk()
            self.layers[i].backward_other_error(ekWk)
        self.update_weights_and_bias_backward(x)

    def update_weights_and_bias_backward(self, x):
        self.layers[0].update_weights_and_bias_backward(x.transpose())
        for i in range(1, self.n_layers):
            self.layers[i].update_weights_and_bias_backward(self.results[i - 1])

    def set_weights_and_bias(self, weights_list, bias_list):
        for i in range(len(weights_list)):
            self.layers[i].weights = weights_list[i]
        for i in range(len(bias_list)):
            self.layers[i].bias = bias_list[i]
