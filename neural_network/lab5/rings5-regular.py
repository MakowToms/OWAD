from neural_network.activations import *
from neural_network.data_manipulation import load_data
from neural_network.plots import plot_data_2d, plot_measure_results_data
from neural_network.learn_network import learn_network

# load dataset
train, test = load_data(file_name='rings5-regular', folder='classification', classification=True)

# x and y - observations and values for them
x = train[:, 0:2]
y = train[:, 2:3]

# plot data classes
plot_data_2d(x[:, 0], x[:, 1], y[:, 0], title='True classes of points on the plane')

neurons = [50, 50, 50]

# learn model and plot result classes for i layer networks
for i in range(1, 4):
    mse_linear = learn_network(x, y, neurons[:i], [linear] * i + [softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with linear activation function and " + str(i) + " hidden layers")
    mse_ReLU = learn_network(x, y, neurons[:i], [ReLU] * i + [softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with ReLU activation function and " + str(i) + " hidden layers")
    mse_sigmoid = learn_network(x, y, neurons[:i], [sigmoid] * i + [softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with sigmoid activation function and " + str(i) + " hidden layers")
    mse_tanh = learn_network(x, y, neurons[:i], [tanh] * i + [softmax], beta=0.01, eta=0.01, epochs=1, iterations=1000, regression=False, plot_title="Prediction with tanh activation function and " + str(i) + " hidden layers")

    plot_measure_results_data([mse_linear, mse_ReLU, mse_sigmoid, mse_tanh], labels=['linear', 'ReLU', 'sigmoid', 'tanh'], title_base="accuracy", title_ending=" for " + str(i) + " layers networks")

# Results
# The best activation is tanh, then (ReLU and sigmoid)*, linear is the worst one
# * for bigger networks sigmoid is better (and as good as tanh) than ReLU and for 3 layers ReLU had nan in gradient and then it fails
# Linear activation is the worst for any number of layers
# The more layers the better results achieve networks
