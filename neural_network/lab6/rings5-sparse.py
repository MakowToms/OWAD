from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.plots import plot_data_2d, plot_measure_results_data
from neural_network.learn_network import learn_network
from neural_network.testing_model import add_the_same_value_to_the_same_length
import matplotlib.pyplot as plt

# load dataset
train, test = load_data(file_name='rings5-sparse', folder='classification', classification=True)

# x and y - observations and values for them
x = train[:, 0:2]
y = train[:, 2:3]
x_test = test[:, 0:2]
y_test = test[:, 2:3]

# plot data classes
plt.figure(figsize=(12.8, 4.8))
plt.subplot(121)
plot_data_2d(x[:, 0], x[:, 1], y[:, 0], title='True classes of points from rings5 sparse dataset', show=False)

neurons = [50, 50, 50]

labels = ['no regularization', 'L1, lambda: 0.001', 'L1, lambda: 0.0001', 'L1, lambda: 0.00001', 'L2, lambda: 0.001', 'L2, lambda: 0.0001', 'L2, lambda: 0.00001']

# learn model and plot result classes
i = 2
activation = sigmoid
iterations = 10000
no_change_epochs_to_stop = 100
mse_no_reg = learn_network(x, y, neurons[:i], [activation] * i + [softmax], beta=0.01, eta=0.01, epochs=1, iterations=iterations, regression=False, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers", plot=True, return_result=False, x_test=x_test, y_test=y_test, use_test_and_stop_learning=True, no_change_epochs_to_stop=no_change_epochs_to_stop)
mses = [mse_no_reg]
for reg_type in ["L1", "L2"]:
    for reg_lambda in [0.001, 0.0001, 0.00001]:
        mse_reg = learn_network(x, y, neurons[:i], [activation] * i + [softmax], beta=0.01, eta=0.01, epochs=1, iterations=iterations, regression=False, regularization_lambda=reg_lambda, regularization_type=reg_type, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers", plot=True, return_result=False, x_test=x_test, y_test=y_test, use_test_and_stop_learning=True, no_change_epochs_to_stop=no_change_epochs_to_stop)
        mses.append(mse_reg)

# plot from 5th iteration to better see differences
measures = add_the_same_value_to_the_same_length(mses)
plt.subplot(122)
plot_measure_results_data(measures, labels=labels, title_base="accuracy", ylabel="accuracy", title_ending=" for different regularization methods", from_error=200)
plt.show()
