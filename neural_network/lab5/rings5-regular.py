from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.plots import plot_data_2d, plot_measure_results_data
from neural_network.learn_network import learn_network

# load dataset
train, test = load_data(file_name='rings5-regular', folder='classification', classification=True)

# x and y - observations and values for them
x = train[:, 0:2]
y = train[:, 2:3]

# plot data classes
plot_data_2d(x[:, 0], x[:, 1], y[:, 0], title='True classes of points on the plane')

# learn model and plot result classes
mse_linear = learn_network(x, y, [50, 50], [linear, linear, softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with linear activation function")
mse_ReLU = learn_network(x, y, [50, 50], [ReLU, ReLU, softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with ReLU activation function")
mse_sigmoid = learn_network(x, y, [50, 50], [sigmoid, sigmoid, softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with sigmoid activation function")
mse_tanh = learn_network(x, y, [50, 50], [tanh, tanh, softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with tanh activation function")

plot_measure_results_data([mse_linear, mse_ReLU, mse_sigmoid, mse_tanh], labels=['linear', 'ReLU', 'sigmoid', 'tanh'])
