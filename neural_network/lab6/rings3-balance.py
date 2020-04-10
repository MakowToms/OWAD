from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.plots import plot_data_2d, plot_measure_results_data
from neural_network.learn_network import learn_network

# load dataset
train, test = load_data(file_name='rings3-balance', folder='classification', classification=True)

# x and y - observations and values for them
x = train[:, 0:2]
y = train[:, 2:3]

# plot data classes
plot_data_2d(x[:, 0], x[:, 1], y[:, 0], title='True classes of points on the plane')

neurons = [50, 50, 50]

# learn model and plot result classes
i = 2
activation = sigmoid
iterations = 1000
mse_no_reg = learn_network(x, y, neurons[:i], [activation] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=iterations, regression=False, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers")
mse_reg_001 = learn_network(x, y, neurons[:i], [activation] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=iterations, regression=False, regularization_lambda=0.01, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers")
mse_reg_0001 = learn_network(x, y, neurons[:i], [activation] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=iterations, regression=False, regularization_lambda=0.001, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers")
mse_reg_00001 = learn_network(x, y, neurons[:i], [activation] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=iterations, regression=False, regularization_lambda=0.0001, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers")
mse_reg_000001 = learn_network(x, y, neurons[:i], [activation] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=iterations, regression=False, regularization_lambda=0.00001, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers")

plot_measure_results_data([mse_no_reg, mse_reg_0001, mse_reg_00001, mse_reg_000001], labels=['no regularization', '-3', '-4', '-5'], title_base="accuracy", title_ending=" for " + str(i) + " layers networks")
