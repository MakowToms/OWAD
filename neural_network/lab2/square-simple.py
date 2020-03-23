import matplotlib.pyplot as plt

from neural_network.Network import Network
from neural_network.activations import *
from neural_network.loading_data import load_data
from neural_network.testing_model import MSE

train, test = load_data('square-simple')

x = train[:, 0:1]
y = train[:, 1:2]

network = Network(1, [10], 1, [sigmoid, linear], initialize_weights='norm')

plt.figure()
plt.plot(x[:, 0], y[:, 0], 'bo')

for i in range(50):
    res = network.forward(x)
    print('Iteracja {0}, MSE: {1:0.8f}'.format(i, MSE(res, y)))
    network.backward(x, y, eta=0.1, epochs=100)

plt.plot(x[:, 0], res[:, 0], 'bo')
plt.show()
