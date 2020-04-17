from neural_network.activations import *
from neural_network.data_manipulation import load_easy_data
from neural_network.plots import plot_data_2d, plot_measure_results_data
from neural_network.learn_network import learn_network
import matplotlib.pyplot as plt

# load dataset
train, test = load_easy_data()

# x and y - observations and values for them
x = train[:, 0:2]
y = train[:, 2:3]

# plot data classes
plt.figure(figsize=(12.8, 9.6))
plt.subplot(221)
plot_data_2d(x[:, 0], x[:, 1], y[:, 0], title='True classes of points on the plane', show=False)

# learn model and plot result classes
plt.subplot(222)
mse_linear = learn_network(x, y, [20], [sigmoid, linear], iterations=100, regression=False, plot_title='Predicted classes for linear function', plot_show=False)
plt.subplot(223)
mse_softmax = learn_network(x, y, [20], [sigmoid, softmax], iterations=100, regression=False, plot_title='Predicted classes for softmax function', plot_show=False)

plt.subplot(224)
plot_measure_results_data([mse_linear, mse_softmax], labels=['linear', 'softmax'], title_base="Accuracy", ylabel="Accuracy", title_ending=" for last layer activation function", show=False)
plt.show()
