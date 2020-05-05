import pandas as pd
import numpy as np


def normalize(X):
    return (X-np.mean(X, 0))/np.std(X, 0)


def load_data(file_name, normalization=True):
    data = pd.read_csv('kohonen/data/' + file_name + '.csv')
    n = data.shape[1]
    classes = np.array(data.iloc[:, n-1])
    data = np.array(data.iloc[:, :(n-1)])
    if normalization:
        data = normalize(data)
    return data, classes


def load_mnist(train_test="train"):
    data = pd.read_csv("kohonen/data/mnist_" + train_test + ".csv")
    classes = np.array(data.iloc[:, 0])
    data = np.array(data.iloc[:, 1:] / 256)
    return data, classes


def load_har(train_test="train"):
    file = "kohonen/data/UCI HAR Dataset/" + train_test + "/X_" + train_test + ".txt"
    data = pd.read_csv(file, sep="\s+", engine='python')
    classes = pd.read_csv("kohonen/data/UCI HAR Dataset/" + train_test + "/y_" + train_test + ".txt")
    classes = np.array(classes)
    data = np.array(data)
    return data, classes
