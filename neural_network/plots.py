import matplotlib.pyplot as plt
import numpy as np


def plot_data_2d(x, y, colors):
    plt.figure()
    uniques = np.unique(colors)
    different_colors = ['r', 'g', 'b', 'y', 'o', 'c', 'k']
    for index, unique in enumerate(uniques):
        indexes = colors == unique
        plt.plot(x[indexes], y[indexes], different_colors[index] + 'o')
    plt.show()


def plot_mse_data(mse, labels=["base model", "momentum model", "RMSProp model"]):
    plt.figure()
    plt.plot(mse[0], color='red')
    plt.plot(mse[1], color='green')
    plt.plot(mse[2], color='blue')
    plt.legend(labels=labels)
    plt.show()


def plot_data_1d(x, true, prediction):
    plt.figure()
    plt.plot(x[:, 0], true[:, 0], 'bo')
    plt.plot(x[:, 0], prediction[:, 0], 'go')
    plt.legend(labels=["true data", "prediction"])
    plt.show()
