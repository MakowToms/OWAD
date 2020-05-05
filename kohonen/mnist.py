from kohonen.data_manipulation import load_mnist
from kohonen.Kohonen import Kohonen
import matplotlib.pyplot as plt
from kohonen.plots import plot_data_2d

# to load data it is needed to download it from
# https://www.kaggle.com/oddrationale/mnist-in-csv
# and put 2 files in data: data/mnist_train.csv data/mnist_test.csv
x, y = load_mnist("train")
x_test, y_test = load_mnist("test")

# check gauss parameters and plot it
plt.figure(figsize=(6.4 * 2, 4.8 * 2))
for i, neighbour in enumerate([0.2, 0.5]):
    for j, grid_type in enumerate(["square", "hexagonal"]):
        size = 5
        network = Kohonen(size, size, x.shape[1], neighbour_param=neighbour, distance_type="gauss", grid_type=grid_type, use_pca=True)
        network.learn_epochs(x, epochs=10)

        x_to_plot = network.transform_with_pca(x_test)
        x_to_plot = x_to_plot[:1000]
        # plot network
        plt.subplot(2, 2, 1 + i + 2*j)
        title = "Neighbour: " + str(neighbour) + ", metric: Gauss, grid type: " + grid_type
        plot_data_2d(x_to_plot[:, :1], x_to_plot[:, 1:], y_test[:1000], show=False, title=title)
        network.plot_weights()
plt.show()

# check mexican hat parameters and plot it
plt.figure(figsize=(6.4 * 2, 4.8 * 2))
for i, neighbour in enumerate([0.01, 0.015]):
    for j, grid_type in enumerate(["square", "hexagonal"]):
        size = 5
        network = Kohonen(size, size, x.shape[1], neighbour_param=neighbour, distance_type="mexican_hat", grid_type=grid_type,
                          use_pca=True)
        network.learn_epochs(x, epochs=10)

        x_to_plot = network.transform_with_pca(x_test)
        x_to_plot = x_to_plot[:1000]
        # plot network
        plt.subplot(2, 2, 1 + i + 2 * j)
        title = "Neighbour: " + str(neighbour) + ", metric: mexican hat, grid type: " + grid_type
        plot_data_2d(x_to_plot[:, :1], x_to_plot[:, 1:], y_test[:1000], show=False, title=title)
        network.plot_weights()
plt.show()
