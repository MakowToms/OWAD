from neural_network.Network import Network
from neural_network.activations import *
from neural_network.data_manipulation import load_data, one_hot_encode
from neural_network.testing_model import accuracy
from neural_network.plots import plot_data_2d
from neural_network.learn_network import learn_network

# load dataset
train, test = load_data(file_name='xor3', folder='classification', classification=True)

# get x, y
x = train[:, 0:2]
y = train[:, 2:3]

# plot data classes
plot_data_2d(x[:, 0], x[:, 1], y[:, 0])

# learn model and plot result classes
mse_rms = learn_network(x, y, [50, 50], [sigmoid, sigmoid, softmax], beta=0.05, eta=0.01, epochs=1, iterations=100, regression=False)
