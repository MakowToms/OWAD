import matplotlib.pyplot as plt
import numpy as np


# visualization of 2d data
def plot_data_2d(x, y, colors, title='Points on the plane', ylabel='y', xlabel='x'):
    plt.figure()
    uniques = np.unique(colors)
    different_colors = ['r', 'g', 'b', 'y', 'c', 'k']
    labels = []
    for index, unique in enumerate(uniques):
        indexes = colors == unique
        plt.plot(x[indexes], y[indexes], different_colors[index] + 'o')
        labels.append('class ' + str(int(unique)))
    plt.legend(labels=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# plot which shows mse or accuracy
def plot_measure_results_data(errors, title_base='MSE', title_ending=' through epochs', labels=["base model", "momentum model", "RMSProp model"], colors=['red', 'green', 'blue', 'yellow'], xlabel='epochs', ylabel='MSE'):
    plt.figure()
    for i in range(len(errors)):
        plt.plot(errors[i], color=colors[i])
    plt.legend(labels=labels)
    plt.title(title_base + title_ending)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# visualization of 1d data with both predictions and true values
def plot_data_1d(x, true, prediction, title='Fictional data with values', ylabel='result values', xlabel='observed values'):
    plt.figure()
    plt.plot(x[:, 0], true[:, 0], 'bo')
    plt.plot(x[:, 0], prediction[:, 0], 'go')
    plt.legend(labels=["true data", "prediction"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_data_1d_compare(x, true, predictions, title='Fictional data with values', ylabel='result values', xlabel='observed values', labels=["true data", "prediction"]):
    plt.figure()
    plt.plot(x[:, 0], true[:, 0], 'bo')
    different_colors = ['r', 'g', 'y', 'c', 'k']
    for index, prediction in enumerate(predictions):
        plt.plot(x[:, 0], prediction[:, 0], different_colors[index] + 'o')
    plt.legend(labels=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
