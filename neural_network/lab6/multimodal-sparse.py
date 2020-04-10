from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.plots import plot_data_1d_compare, plot_measure_results_data
from neural_network.learn_network import learn_network

# load dataset
train, test = load_data('multimodal-sparse')

# x and y - observations and values for them
x = train[:, 0:1]
y = train[:, 1:2]

# neurons in each layer
neurons = [50, 50, 50]
# labels for plots
labels = ['no regularization', '0.0001', '0.01', '0.001']

# learn model and plot result classes
i = 2
activation = sigmoid
iterations = 10000
mse_no_reg, res_no_reg = learn_network(x, y, neurons[:i], [activation] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=iterations, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers", plot=False, return_result=True)
mse_reg_00001, res_reg_00001 = learn_network(x, y, neurons[:i], [activation] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=iterations, regularization_lambda=0.0001, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers", plot=False, return_result=True)
mse_reg_001, res_reg_001 = learn_network(x, y, neurons[:i], [activation] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=iterations, regularization_lambda=0.01, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers", plot=False, return_result=True)
mse_reg_0001, res_reg_0001 = learn_network(x, y, neurons[:i], [activation] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=iterations, regularization_lambda=0.001, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers", plot=False, return_result=True)

plot_data_1d_compare(x, y, [res_no_reg, res_reg_00001, res_reg_001, res_reg_0001], labels=["true"] + labels, title="Comparison of activation functions for " + str(i) + " hidden layers networks")
# plot from 5th iteration to better see differences
plot_measure_results_data([mse_no_reg[5:], mse_reg_00001[5:], mse_reg_001[5:], mse_reg_0001[5:]], labels=labels, title_ending=" for " + str(i) + " layers networks")
