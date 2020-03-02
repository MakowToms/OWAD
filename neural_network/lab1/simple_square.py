from numpy import genfromtxt
import matplotlib.pyplot as plt

from neural_network.Network import Network
from neural_network.activations import *


def MSE(x, y):
    return np.mean(x - y) ** 2


# simple square

train = genfromtxt('mio1/regression/square-simple-training.csv', delimiter=',')
train = train[1:, 1:]
test = genfromtxt('mio1/regression/square-simple-test.csv', delimiter=',')
test = test[1:, 1:]

x = train[:, 0:1]
y = train[:, 1:2]
x = x.transpose()
y = y.transpose()

# fig = plt.figure()
plt.plot(x[0, :], y[0, :], 'bo')

# base architecture

simple_square = Network(1, [5], 1, [sigmoid, linear])
res = simple_square.forward(x)
print(MSE(res, y))

simple_square.set_weights_and_bias([np.array([[5], [-2], [1], [1], [1]]), np.array([[1, 1, 1, 1, 1]])], [np.array([[1], [0], [0], [0], [1]]), np.array([[1]])])
res = simple_square.forward(x)
print(MSE(res, y))
plt.plot(x[0, :], res[0, :], 'bo')

# other architectures

simple_square2 = Network(1, [10], 1, [sigmoid, linear])
res = simple_square2.forward(x)
print(MSE(res, y))

simple_square3 = Network(1, [5, 5], 1, [sigmoid, sigmoid, linear])
res = simple_square3.forward(x)
print(MSE(res, y))


