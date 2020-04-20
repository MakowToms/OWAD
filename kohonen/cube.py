from kohonen.data_manipulation import load_data
from kohonen.Kohonen import Kohonen

data, classes = load_data("cube")
network = Kohonen(10, 10, 3)

network.learn_epochs(data)

network.weights
