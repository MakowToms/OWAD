from neural_network.activations import *
from neural_network.learn_network import learn_network
from neural_network.data_manipulation import load_data
from neural_network.plots import plot_data_1d_compare, plot_measure_results_data


train, test = load_data('steps-small')

x = train[:, 0:1]
y = train[:, 1:2]

# weights experiment
mse_zeros, zeros = learn_network(x, y, [20], [sigmoid, linear], initialize_weights='zeros', iterations=200, momentum_type='normal', plot=False, return_result=True)
mse_normal, normal = learn_network(x, y, [20], [sigmoid, linear], initialize_weights='norm', iterations=200, momentum_type='normal', plot=False, return_result=True)
mse_Xavier, Xavier = learn_network(x, y, [20], [sigmoid, linear], initialize_weights='Xavier', iterations=200, momentum_type='normal', plot=False, return_result=True)

labels = ['Zero weights', 'Uniform distribution', 'Xavier']

plot_data_1d_compare(x, y, [zeros, normal, Xavier], labels=["true"] + labels, title="Comparison of initialization weights")
plot_measure_results_data([mse_zeros, mse_normal, mse_Xavier], labels=labels, title_ending=" for initialization weights", from_error=5)

# batch size experiment
mse_all, res_all = learn_network(x, y, [20], [sigmoid, linear], initialize_weights='Xavier', eta=0.01, batch_size=10000, iterations=100, momentum_type='normal', plot=False, return_result=True)
mse_batch, batch = learn_network(x, y, [20], [sigmoid, linear], initialize_weights='Xavier', eta=0.01, batch_size=32, iterations=100, momentum_type='normal', plot=False, return_result=True)
mse_mini_batch, mini_batch = learn_network(x, y, [20], [sigmoid, linear], initialize_weights='Xavier', eta=0.01, batch_size=4, iterations=100, momentum_type='normal', plot=False, return_result=True)

labels = ['all observations together', 'batch of 32 observations', 'batch of 4 observations']
plot_data_1d_compare(x, y, [res_all, batch, mini_batch], labels=["true"] + labels, title="Comparison of types of learning")
plot_measure_results_data([mse_all, mse_batch, mse_mini_batch], labels=labels, title_ending=" for types of learning", from_error=10)
