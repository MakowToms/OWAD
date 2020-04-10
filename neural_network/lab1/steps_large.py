import matplotlib.pyplot as plt

from neural_network.Network import Network
from neural_network.activations import *

from neural_network.data_manipulation import load_data
from neural_network.testing_model import MSE

train, test = load_data('steps-large')

x = train[:, 0:1]
y = train[:, 1:2]

# base architecture
network = Network(1, [5], 1, [sigmoid, linear])
res = network.forward(x)
print(MSE(res, y))

network.set_weights_and_bias([np.array([[50], [50], [0], [-50], [-50]]), np.array([[0.95, 0.7, 0, -0.3, -0.95]])], [np.array([[-60], [-10], [0], [10], [-40]]), np.array([[0]])])
res = network.forward(x)
print(MSE(res, y))
plt.plot(x, y, 'bo')
plt.plot(x, res, 'ro')
plt.title("Neural network with weights and bias written by hand")
plt.legend(["true", "predicted"])
plt.xlabel('observed values')
plt.ylabel('result values')
plt.savefig("lab1_steps-large.png")
plt.show()
# other architectures - works

# network2 = Network(1, [10], 1, [sigmoid, linear])
# res = network2.forward(x)
# print(MSE(res, y))
#
# network3 = Network(1, [5, 5], 1, [sigmoid, sigmoid, linear])
# res = network3.forward(x)
# print(MSE(res, y))


