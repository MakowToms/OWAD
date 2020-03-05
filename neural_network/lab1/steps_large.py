from numpy import genfromtxt
import matplotlib.pyplot as plt

from neural_network.Network import Network
from neural_network.activations import *


def MSE(x, y):
    return np.mean(x - y) ** 2


# steps large

train = genfromtxt('mio1/regression/steps-large-training.csv', delimiter=',')
train = train[1:, 1:]
test = genfromtxt('mio1/regression/steps-large-test.csv', delimiter=',')
test = test[1:, 1:]

x = train[:, 0:1]
y = train[:, 1:2]
x.shape
# x = x.transpose()
# y = y.transpose()

fig = plt.figure()
plt.plot(x[:, 0], y[:, 0], 'bo')

# base architecture

steps_large = Network(1, [5], 1, [sigmoid, linear])
res = steps_large.forward(x)
print(MSE(res, y))

for i in range(y.shape[1]):
    steps_large.backward(x, y)

plt.plot(x[:, 0], res[:, 0], 'bo')

steps_large.set_weights_and_bias([np.array([[5], [-2], [1], [1], [1]]), np.array([[1, 1, 1, 1, 1]])], [np.array([[0], [0], [0], [0], [0]]), np.array([[0]])])
res = steps_large.forward(x)
print(MSE(res, y))

# other architectures

steps_large2 = Network(1, [10], 1, [sigmoid, linear])
res = steps_large2.forward(x)
print(MSE(res, y))

steps_large3 = Network(1, [5, 5], 1, [sigmoid, sigmoid, linear])
res = steps_large3.forward(x)
print(MSE(res, y))


