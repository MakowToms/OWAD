import matplotlib.pyplot as plt
import numpy as np


# visualization of 2d data
def plot_data_2d(x, y, colors, title='Points on the plane', ylabel='y', xlabel='x', show=True):
    uniques = np.unique(colors)
    different_colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'purple', 'brown', 'orange']
    labels = []
    for index, unique in enumerate(uniques):
        indexes = colors == unique
        plt.plot(x[indexes], y[indexes], 'o', color=different_colors[index])
        labels.append('class ' + str(int(unique)))
    plt.legend(labels=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()
