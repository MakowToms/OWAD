import matplotlib.pyplot as plt

from neural_network.Network import Network
from neural_network.activations import *

from neural_network.data_manipulation import load_data
from neural_network.testing_model import MSE

train, test = load_data('square-simple')

x = train[:, 0:1]
y = train[:, 1:2]

# base architecture
simple_square = Network(1, [5], 1, [sigmoid, linear])
res = simple_square.forward(x)
print(MSE(res, y))

simple_square.set_weights_and_bias([np.array([[15], [5], [10], [-8], [-10]]), np.array([[1.5, -0.5, 2, -0.8, 2]])], [np.array([[-12], [1], [-15], [1], [-15]]), np.array([[0]])])
res = simple_square.forward(x)
print(MSE(res, y))
plt.plot(x, y, 'bo')
plt.plot(x, res, 'ro')
plt.title("Neural network with weights and bias written by hand")
plt.legend(["true", "predicted"])
plt.xlabel('observed values')
plt.ylabel('result values')
plt.savefig("lab1_simple-square.png")
plt.show()

# other architectures - works

# simple_square2 = Network(1, [10], 1, [sigmoid, linear])
# res = simple_square2.forward(x)
# print(MSE(res, y))
#
# simple_square3 = Network(1, [5, 5], 1, [sigmoid, sigmoid, linear])
# res = simple_square3.forward(x)
# print(MSE(res, y))


