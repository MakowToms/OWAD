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

train = normalize(train)

x = train[:, 0:1]
y = train[:, 1:2]
x = x.transpose()
y = y.transpose()

simple_square = Network(1, [5], 1, [sigmoid, linear])
# simple_square.backward(x[:1, :1], y[:1, :1])

for i in range(y.shape[1]):
    simple_square.backward(x[:1, i:i+1], y[:1, i:i+1])

a = simple_square.layers[0].backward_error
b = simple_square.layers[0].weights
np.matmul(a, b.transpose())

simple_square.layers[1].backward_error

simple_square.layers[0].weights
simple_square.layers[0].bias

def MSE(x, y):
    return np.mean(x - y) ** 2
plt.plot(x[0, :], y[0, :], 'bo')
res = simple_square.forward(x)
print(MSE(res, y))
plt.plot(x[0, :], res[0, :], 'bo')

