from neural_network.Layer import Layer


class Network:
    def __init__(self, inp, hidden, out, activations):
        if hidden is None:
            hidden = []
        if len(hidden) + 1 != len(activations):
            print('Bad sizes')

        layers = []
        hidden.insert(0, inp)
        hidden.append(out)
        for i in range(len(activations)):
            layers.append(Layer(hidden[i], hidden[i+1], activations[i]))
        self.layers = layers
        self.n_layers = len(layers)

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i].forward(x)
        return x

    def set_weights_and_bias(self, weights_list, bias_list):
        for i in range(len(weights_list)):
            self.layers[i].weights = weights_list[i]
        for i in range(len(bias_list)):
            self.layers[i].bias = bias_list[i]
