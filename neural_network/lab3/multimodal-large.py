from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.learn_network import learn_network
from neural_network.plots import plot_measure_results_data, plot_data_1d_compare

# load dataset
train, test = load_data('multimodal-large')

x = train[:, 0:1]
y = train[:, 1:2]

# learn network without any momentum technique and show model
mse, base = learn_network(x, y, [20], [sigmoid, linear], momentum_type='normal', eta=0.01, epochs=1, iterations=100, plot=False, return_result=True)

# learn network with momentum and show model
mse_mom, mom = learn_network(x, y, [20], [sigmoid, linear], momentum_type='momentum', lambda_momentum=0.9, eta=0.01, epochs=1, iterations=100, plot=False, return_result=True)

# learn model with RMSProp and show model
mse_rms, rms = learn_network(x, y, [20], [sigmoid, linear], momentum_type='RMSProp', beta=0.01, eta=0.1, epochs=1, iterations=100, plot=False, return_result=True)

labels = ['No momentum', 'Momentum', 'RMSProp']

# plot data and mse
plot_data_1d_compare(x, y, [base, mom, rms], labels=["true"] + labels, title="Comparison of momentum learning")
plot_measure_results_data([mse, mse_mom, mse_rms], labels=labels, title_ending=" for momentum learning", from_error=5)
