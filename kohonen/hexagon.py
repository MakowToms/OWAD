from kohonen.data_manipulation import load_data
from kohonen.Kohonen import Kohonen
from kohonen.plot import plot_point_map

data, classes = load_data("hexagon")

network = Kohonen(15, 15, 2, neighbour_param=0.8, distance_type="gauss", t=1)
network.learn_epochs(data, epochs=50)
plot_point_map(data, classes, network, ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], colors=['r', 'g', 'b', 'y', 'c', 'k'])

# network.weights[:, :, 1]
#
# from neural_network.plots import plot_data_2d
# plot_data_2d(data[:, :1], data[:, 1:], classes)

#Mexican hat
network = Kohonen(15, 15, 2, neighbour_param=0.8, distance_type="not_gauss_(its_enough)", t=1)
network.learn_epochs(data, epochs=50)
plot_point_map(data, classes, network, ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], colors=['r', 'g', 'b', 'y', 'c', 'k'])


network = Kohonen(10, 10, 2, neighbour_param=0.1, distance_type="not_gauss_(its_enough)", t=0.01)
network.learn_epochs(data, epochs=50)
plot_point_map(data, classes, network, ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], colors=['r', 'g', 'b', 'y', 'c', 'k'])
