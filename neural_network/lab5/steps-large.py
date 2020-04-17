from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.plots import plot_data_1d_compare, plot_measure_results_data
from neural_network.learn_network import learn_network
import matplotlib.pyplot as plt

# load dataset
train, test = load_data('steps-large')

# x and y - observations and values for them
x = train[:, 0:1]
y = train[:, 1:2]

# neurons in each layer
neurons = [50, 50, 50]
# labels for plots
labels = ['linear', 'ReLU', 'sigmoid', 'tanh']

# learn model and plot result classes
plt.figure(figsize=(12.8, 19.2))
for i in range(1, 4):
    mse_linear, res_linear = learn_network(x, y, neurons[:i], [linear] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=300, plot_title="Prediction with linear activation function and " + str(i) + " hidden layers", plot=False, return_result=True)
    mse_ReLU, res_ReLU = learn_network(x, y, neurons[:i], [ReLU] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=300, plot_title="Prediction with ReLU activation function and " + str(i) + " hidden layers", plot=False, return_result=True)
    mse_sigmoid, res_sigmoid = learn_network(x, y, neurons[:i], [sigmoid] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=300, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers", plot=False, return_result=True)
    mse_tanh, res_tanh = learn_network(x, y, neurons[:i], [tanh] * i + [linear], beta=0.01, eta=0.01, epochs=1, iterations=300, plot_title="Prediction with tanh activation function and " + str(i) + " hidden layers", plot=False, return_result=True)

    plt.subplot(420 + 2*i - 1)
    plot_data_1d_compare(x, y, [res_linear, res_ReLU, res_sigmoid, res_tanh], labels=["true"] + labels, title="Comparison of activation functions for " + str(i) + " hidden layers networks", show=False)
    # plot from 5th iteration to better see differences
    plt.subplot(420 + 2*i)
    plot_measure_results_data([mse_linear, mse_ReLU, mse_sigmoid, mse_tanh], labels=labels, title_ending=" for " + str(i) + " layers networks", from_error=5, show=False)
    if i == 3:
        plt.subplot(420 + 2*i + 2)
        plot_measure_results_data([mse_linear, mse_sigmoid, mse_tanh], labels=labels[:1] + labels[2:4], colors=['red', 'cyan', 'yellow'], title_ending=" for " + str(i) + " layers networks", from_error=5, show=False)
plt.show()

# Results
# The worst activation function is linear
# For 1 layer networks the best is ReLU, then sigmoid and tanh
# For 1 layer networks only ReLU make steps, others create something like line
# 2 layer networks are needed to see the best from activation functions:
# only linear still don't make steps
# The best was sigmoid function but after some iterations
# At the beginning (after around 10 epochs) ReLU was the best
# tanh was as good as ReLU
# For 3 layers the best was also sigmoid and then tanh, ReLU had NaN in gradient so it was bad
# There are no big difference between 2 and 3 layers in both sigmoid achieves quite the same results
# And 1 layer network was the worst
