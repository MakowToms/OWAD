from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.plots import plot_data_2d
from neural_network.learn_network import learn_network

# load dataset
train, test = load_data(file_name='xor3', folder='classification', classification=True)

# x and y - observations and values for them
x = train[:, 0:2]
y = train[:, 2:3]

# plot data classes
plot_data_2d(x[:, 0], x[:, 1], y[:, 0], title='True classes of points on the plane')

# learn model and plot result classes
mse_rms = learn_network(x, y, [50, 50], [sigmoid, sigmoid, softmax], beta=0.05, eta=0.01, epochs=1, iterations=500, regression=False)
