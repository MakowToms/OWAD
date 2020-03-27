from neural_network.Network import Network
from neural_network.activations import *
from neural_network.testing_model import MSE, accuracy
from neural_network.data_manipulation import one_hot_encode
from neural_network.plots import plot_data_1d, plot_data_2d


# function to create network, learn, return error and plot prediction
def learn_network(x, y, hidden_layers, activations, initialize_weights='Xavier', momentum_type='RMSProp', lambda_momentum=0.9, beta=0.01, eta=0.01, epochs=1, iterations=20, regression=True, plot_title=None, return_result=False, plot=True):
    # for classification problem we need one hot encoded y
    if regression:
        y_numeric = -1
    else:
        y_numeric = y
        y = one_hot_encode(y)

    # determine network parameters and initialize it
    n_input = x.shape[1]
    n_output = y.shape[1]
    network = Network(n_input, hidden_layers, n_output, activations, initialize_weights=initialize_weights)

    # initialize error list, predict and save default error
    errors = []
    res = network.forward(x)
    append_errors(errors, res, y, y_numeric, regression)

    # in for loop train network
    for i in range(iterations):
        # train network (with backward propagation) depends on type of training (nothing, momentum, RMSProp)
        if momentum_type == 'normal':
            network.backward(x, y, eta=eta, epochs=epochs)
        if momentum_type == 'momentum':
            network.backward(x, y, eta=eta, epochs=epochs, lambda_momentum=lambda_momentum, moment_type='momentum')
        if momentum_type == 'RMSProp':
            network.backward(x, y, eta=eta, epochs=epochs, beta=beta, moment_type='RMSProp')

        # predict and save error after i*epochs epochs
        res = network.forward(x)
        append_errors(errors, res, y, y_numeric, regression)
        measure_name = "MSE" if regression else "accuracy"
        print('Iteration {0}, {1}: {2:0.8f}'.format(i, measure_name, errors[-1]))

    # plot data after all training
    if plot:
        if regression:
            if plot_title is not None:
                plot_data_1d(x, y, res, title=plot_title)
            else:
                plot_data_1d(x, y, res)
        else:
            res = np.argmax(res, axis=1)
            if plot_title is not None:
                plot_data_2d(x[:, 0], x[:, 1], res, title=plot_title)
            else:
                plot_data_2d(x[:, 0], x[:, 1], res, title='Predicted classes of points on the plane')
    if return_result:
        return errors, res
    return errors


# function to write next error to list
def append_errors(errors, res, y, y_numeric, regression):
    if regression:
        errors.append(MSE(res, y))
    else:
        res = np.argmax(res, axis=1)
        errors.append(accuracy(res, y_numeric))
    return errors
