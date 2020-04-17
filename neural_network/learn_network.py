from neural_network.Network import Network
from neural_network.activations import *
from neural_network.testing_model import MSE, accuracy
from neural_network.data_manipulation import one_hot_encode
from neural_network.plots import plot_data_1d, plot_data_2d
import math

# function to create network, learn, return error and plot prediction
def learn_network(x, y, hidden_layers, activations, x_test=None, y_test=None, use_test_and_stop_learning=False, no_change_epochs_to_stop=50, initialize_weights='norm', momentum_type='RMSProp', lambda_momentum=0.9, beta=0.01, eta=0.01, epochs=1, batch_size=32, iterations=20, regularization_lambda=0, regularization_type="L2", regression=True, plot_title=None, return_result=False, plot=True, plot_show=True):
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
    # if use test dataset and regularization should test on test dataset
    if use_test_and_stop_learning:
        res = network.forward(x_test)
        append_errors(errors, res, y_test, one_hot_encode(y_test), regression)
    # else can test on train dataset
    else:
        res = network.forward(x)
        append_errors(errors, res, y, y_numeric, regression)

    # values needed for have in memory best network result and information how many epochs ago was positive change
    best_result = set_up_best_result(regression)
    any_change = 0
    # in for loop train network
    for i in range(iterations):
        # train network (with backward propagation) depends on type of training (nothing, momentum, RMSProp)
        if momentum_type == 'normal':
            network.backward(x, y, eta=eta, epochs=epochs, batch_size=batch_size, regularization_lambda=regularization_lambda, regularization_type=regularization_type)
        if momentum_type == 'momentum':
            network.backward(x, y, eta=eta, epochs=epochs, batch_size=batch_size, regularization_lambda=regularization_lambda, regularization_type=regularization_type, lambda_momentum=lambda_momentum, moment_type='momentum')
        if momentum_type == 'RMSProp':
            network.backward(x, y, eta=eta, epochs=epochs, batch_size=batch_size, regularization_lambda=regularization_lambda, regularization_type=regularization_type, beta=beta, moment_type='RMSProp')

        # predict and save error after i*epochs epochs
        res = network.forward(x)
        append_errors(errors, res, y, y_numeric, regression)
        measure_name = "MSE" if regression else "accuracy"
        print('Iteration {0}, {1}: {2:0.8f}'.format(i, measure_name, errors[-1]))
        # determine if should stop learning because of any better result in last 10 epochs
        if use_test_and_stop_learning:
            should_stop, best_result, any_change = stop_learning(errors[-1], best_result, any_change, regression, no_change_epochs_to_stop=no_change_epochs_to_stop)
            if should_stop:
                break

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
                plot_data_2d(x[:, 0], x[:, 1], res, title=plot_title, show=plot_show)
            else:
                plot_data_2d(x[:, 0], x[:, 1], res, title='Predicted classes of points on the plane', show=plot_show)
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

# set up best result variable based if it's regression
def set_up_best_result(regression):
    if regression:
        return math.inf
    else:
        return 0

# determine if should stop learning because no better results last 10 epochs
def stop_learning(error, best_result, any_change, regression, no_change_epochs_to_stop=50):
    if regression:
        if error < best_result:
            return False, error, 0
        else:
            if any_change < no_change_epochs_to_stop:
                return False, best_result, any_change + 1
            else:
                return True, best_result, any_change + 1
    else:
        # in this case accuracy
        if error > best_result:
            return False, error, 0
        else:
            if any_change < no_change_epochs_to_stop:
                return False, best_result, any_change + 1
            else:
                return True, best_result, any_change + 1
