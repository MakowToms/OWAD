from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.plots import plot_data_1d_compare, plot_measure_results_data
from neural_network.learn_network import learn_network

# load dataset
train, test = load_data('multimodal-large')

# x and y - observations and values for them
x = train[:, 0:1]
y = train[:, 1:2]

# learn model and plot result classes
mse_linear, res_linear = learn_network(x, y, [50, 50], [linear, linear, linear], beta=0.01, eta=0.01, epochs=1, iterations=100, plot_title="Prediction with linear activation function", plot=False, return_result=True)
mse_ReLU, res_ReLU = learn_network(x, y, [50, 50], [ReLU, ReLU, linear], beta=0.01, eta=0.01, epochs=1, iterations=100, plot_title="Prediction with ReLU activation function", plot=False, return_result=True)
mse_sigmoid, res_sigmoid = learn_network(x, y, [50, 50], [sigmoid, sigmoid, linear], beta=0.01, eta=0.01, epochs=1, iterations=100, plot_title="Prediction with sigmoid activation function", plot=False, return_result=True)
mse_tanh, res_tanh = learn_network(x, y, [50, 50], [tanh, tanh, linear], beta=0.01, eta=0.01, epochs=1, iterations=100, plot_title="Prediction with tanh activation function", plot=False, return_result=True)

labels = ['linear', 'ReLU', 'sigmoid', 'tanh']
plot_data_1d_compare(x, y, [res_linear, res_ReLU, res_sigmoid, res_tanh], labels=["true"] + labels)
plot_measure_results_data([mse_linear, mse_ReLU, mse_sigmoid, mse_tanh], labels=labels)

