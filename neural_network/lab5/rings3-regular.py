from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.plots import plot_data_2d, plot_measure_results_data
from neural_network.learn_network import learn_network
import matplotlib.pyplot as plt

# load dataset
train, test = load_data(file_name='rings3-regular', folder='classification', classification=True)

# x and y - observations and values for them
x = train[:, 0:2]
y = train[:, 2:3]

# plot data classes
plt.figure(figsize=(12.8, 9.6))
plt.subplot(221)
plot_data_2d(x[:, 0], x[:, 1], y[:, 0], title='True classes of points on the plane', show=False)

neurons = [50, 50, 50]

# learn model and plot result classes
for i in range(1, 4):
    mse_linear = learn_network(x, y, neurons[:i], [linear] * i + [softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with linear activation function and " + str(i) + " hidden layers", plot=False)
    mse_ReLU = learn_network(x, y, neurons[:i], [ReLU] * i + [softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with ReLU activation function and " + str(i) + " hidden layers", plot=False)
    mse_sigmoid = learn_network(x, y, neurons[:i], [sigmoid] * i + [softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers", plot=False)
    mse_tanh = learn_network(x, y, neurons[:i], [tanh] * i + [softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with tanh activation function and " + str(i) + " hidden layers", plot=False)

    plt.subplot(220 + i + 1)
    plot_measure_results_data([mse_linear, mse_ReLU, mse_sigmoid, mse_tanh], labels=['linear', 'ReLU', 'sigmoid', 'tanh'], title_base="accuracy", ylabel="accuracy", title_ending=" for " + str(i) + " layers networks", show=False, from_error=5)
plt.show()

# Results (more than in rings5-regular / not reapiting the same)
# ReLU is the fastest algorithm to learn (is quite good after some iterations)
# sigmoid is slowest to learn but can achieve really good results
# tanh is usually little bit slower than relu and is more stable

# * fastest / slowest means number of epochs to achieve result

