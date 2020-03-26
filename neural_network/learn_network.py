from neural_network.Network import Network
from neural_network.activations import *
from neural_network.testing_model import MSE, accuracy
from neural_network.data_manipulation import one_hot_encode
from neural_network.plots import plot_data_1d, plot_data_2d


def learn_network(x, y, hidden_layers, activations, initialize_weights='Xavier', momentum_type='RMSProp', lambda_momentum=0.9, beta=0.01, eta=0.01, epochs=1, iterations=20, regression=True):
    if regression:
        y_numeric = -1
    else:
        y_numeric = y
        y = one_hot_encode(y)
    n_input = x.shape[1]
    n_output = y.shape[1]
    network = Network(n_input, hidden_layers, n_output, activations, initialize_weights=initialize_weights)
    errors = []
    res = network.forward(x)
    append_errors(errors, res, y, y_numeric, regression)
    for i in range(iterations):
        if momentum_type == 'normal':
            network.backward(x, y, eta=eta, epochs=epochs)
        if momentum_type == 'momentum':
            network.backward(x, y, eta=eta, epochs=epochs, lambda_momentum=lambda_momentum, moment_type='momentum')
        if momentum_type == 'RMSProp':
            network.backward(x, y, eta=eta, epochs=epochs, beta=beta, moment_type='RMSProp')
        res = network.forward(x)
        append_errors(errors, res, y, y_numeric, regression)
        print('Iteracja {0}, measure result: {1:0.8f}'.format(i, errors[-1]))
    if regression:
        plot_data_1d(x, y, res)
    else:
        res = np.argmax(res, axis=1)
        plot_data_2d(x[:, 0], x[:, 1], res)
    return errors


def append_errors(errors, res, y, y_numeric, regression):
    if regression:
        errors.append(MSE(res, y))
    else:
        res = np.argmax(res, axis=1)
        errors.append(accuracy(res, y_numeric))
    return errors
