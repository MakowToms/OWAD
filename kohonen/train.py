from kohonen.plots import plot_data_2d
from kohonen.Kohonen import Kohonen
import matplotlib.pyplot as plt


def find_epochs_and_neighbour(data, classes, M, N, epochs, neighbour_params, **kwargs):
    plt.figure(figsize=(6.4*len(epochs), 4.8*len(neighbour_params)))
    for i, epoch in enumerate(epochs):
        for j, neighbour_param in enumerate(neighbour_params):
            print(epoch, neighbour_param)
            # create network
            network = Kohonen(M, N, data.shape[1], neighbour_param=neighbour_param, **kwargs)
            network.learn_epochs(data, epochs=epoch)

            to_plot = network.transform_with_pca(data)

            # plot network
            plt.subplot(len(neighbour_params), len(epochs), i + j*len(epochs) + 1)
            try:
                plot_data_2d(to_plot[:, :1], to_plot[:, 1:2], classes, show=False, title=str(epoch) + " epochs, neighbour parameter: " + str(neighbour_param))
                network.plot_weights()
            except:
                network.use_pca = False
                network.plot_weights()
                network.use_pca = True
                print()
    plt.show()
