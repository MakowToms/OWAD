from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.plots import plot_mse_data
from neural_network.learn_network import learn_network

# load dataset
train, test = load_data('steps-large')

x = train[:, 0:1]
y = train[:, 1:2]

# learn network without any momentum technique and show model
mse = learn_network(x, y, [20], [sigmoid, linear], momentum_type='normal', eta=0.01, epochs=1, iterations=20)

# learn network with momentum and show model
mse_mom = learn_network(x, y, [20], [sigmoid, linear], momentum_type='momentum', lambda_momentum=0.9, eta=0.01, epochs=1, iterations=20)

# learn model with RMSProp and show model
mse_rms = learn_network(x, y, [50, 50], [sigmoid, sigmoid, linear], momentum_type='RMSProp', beta=0.1, eta=0.1, epochs=1, iterations=20)

# show error (mse) after each epoch
plot_mse_data([mse, mse_mom, mse_rms])
