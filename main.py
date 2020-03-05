from numpy import genfromtxt
import matplotlib.pyplot as plt

from neural_network.Network import Network
from neural_network.activations import *

train = genfromtxt('mio1/regression/square-simple-training.csv', delimiter=',')
train = train[1:, 1:]
test = genfromtxt('mio1/regression/square-simple-test.csv', delimiter=',')
test = test[1:, 1:]


def normalize(X):
    return (X-np.mean(X, 0))/np.std(X, 0)


def MSE(x, y):
    return np.mean(x - y) ** 2


train = normalize(train)

x = train[:, 0:1]
y = train[:, 1:2]

simple_square = Network(1, [1], 1, [sigmoid, linear])

plt.figure()
plt.plot(x[:, 0], y[:, 0], 'bo')

for i in range(10000):
    res = simple_square.forward(x)
    print('Iteracja {0}, MSE: {1:0.4f}'.format(i, MSE(res, y)))
    simple_square.backward(x, y)

plt.plot(x[:, 0], res[:, 0], 'bo')
