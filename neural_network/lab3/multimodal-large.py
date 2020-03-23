import matplotlib.pyplot as plt

from neural_network.Network import Network
from neural_network.activations import *
from neural_network.loading_data import load_data
from neural_network.testing_model import MSE

# load dataset
train, test = load_data('multimodal-large')

x = train[:, 0:1]
y = train[:, 1:2]

# learn network without any momentum technique
network = Network(1, [20], 1, [sigmoid, linear], initialize_weights='Xavier')
mse = []
res = network.forward(x)
mse.append(MSE(res, y))
for i in range(20):
    network.backward(x, y, eta=0.01, epochs=1)
    res = network.forward(x)
    print('Iteracja {0}, MSE: {1:0.8f}'.format(i, MSE(res, y)))
    mse.append(MSE(res, y))

# and show how model predicts
plt.figure()
plt.plot(x[:, 0], y[:, 0], 'bo')
plt.plot(x[:, 0], res[:, 0], 'bo')
plt.show()

# learn network with momentum
network = Network(1, [20], 1, [sigmoid, linear], initialize_weights='Xavier')
mse_mom = []
res = network.forward(x)
mse_mom.append(MSE(res, y))
for i in range(20):
    network.backward(x, y, eta=0.01, epochs=1, lambda_momentum=0.9, moment_type='momentum')
    res = network.forward(x)
    print('Iteracja {0}, MSE: {1:0.8f}'.format(i, MSE(res, y)))
    mse_mom.append(MSE(res, y))

# and show how model predicts
plt.figure()
plt.plot(x[:, 0], y[:, 0], 'bo')
plt.plot(x[:, 0], res[:, 0], 'bo')
plt.show()

# learn model with RMSProp
network = Network(1, [50, 50], 1, [sigmoid, sigmoid, linear], initialize_weights='Xavier')
mse_rms = []
res = network.forward(x)
mse_rms.append(MSE(res, y))
for i in range(20):
    network.backward(x, y, eta=0.1, epochs=1, beta=0.1, moment_type='RMSProp')
    res = network.forward(x)
    print('Iteracja {0}, MSE: {1:0.8f}'.format(i, MSE(res, y)))
    mse_rms.append(MSE(res, y))

# and show how model predicts
plt.figure()
plt.plot(x[:, 0], y[:, 0], 'bo')
plt.plot(x[:, 0], res[:, 0], 'bo')
plt.show()


# show error (mse) after each epoch
plt.figure()
plt.plot(mse, color='red')
plt.plot(mse_mom, color='green')
plt.plot(mse_rms, color='blue')
plt.legend(labels=["base model", "momentum model", "RMSProp model"])
plt.show()
