from numpy import genfromtxt
from neural_network.activations import *


def normalize(X):
    return (X-np.mean(X, 0))/np.std(X, 0)


def load_data(file_name, normalization=True):
    train = genfromtxt('mio1/regression/' + file_name + '-training.csv', delimiter=',')
    train = train[1:, 1:]
    test = genfromtxt('mio1/regression/' + file_name + '-test.csv', delimiter=',')
    test = test[1:, 1:]
    if normalization:
        train = normalize(train)
        test = normalize(test)
    return train, test
