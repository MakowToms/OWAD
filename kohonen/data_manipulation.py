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
