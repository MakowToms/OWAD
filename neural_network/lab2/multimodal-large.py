import matplotlib.pyplot as plt

from neural_network.Network import Network
from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.testing_model import MSE

train, test = load_data('multimodal-large')

x = train[:, 0:1]
y = train[:, 1:2]

network = Network(1, [50, 50], 1, [sigmoid, sigmoid, linear], initialize_weights='norm')

plt.figure()
plt.plot(x[:, 0], y[:, 0], 'bo')

for i in range(20):
    res = network.forward(x)
    print('Iteracja {0}, MSE: {1:0.4f}'.format(i, MSE(res, y)))
    for i in range(100):
        network.backward(x, y)

plt.plot(x[:, 0], res[:, 0], 'bo')
plt.show()
