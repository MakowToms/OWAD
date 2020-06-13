import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def normalize(X):
    return (X-np.mean(X, 0))/np.std(X, 0)


def load_data(file_name, normalization=True):
    data = pd.read_csv('genetic/data/' + file_name, header=None)
    n = data.shape[1]
    classes = np.array(data.iloc[:, n-1])
    data = np.array(data.iloc[:, :(n-1)])
    if normalization:
        data = normalize(data)
    return data, classes


def load_iris():
    train, classes = load_data(file_name='iris.data', normalization=True)

    classes[classes == 'Iris-setosa'] = 0
    classes[classes == 'Iris-versicolor'] = 1
    classes[classes == 'Iris-virginica'] = 2
    classes = classes.astype(np.int)
    return train, classes


def load_adult(normalization=True):
    data = pd.read_csv('genetic/data/adult.data', header=None)
    n = data.shape[1]
    classes = np.array(data.iloc[:, n - 1])
    classes[classes == ' <=50K'] = 0
    classes[classes == ' >50K'] = 1
    classes = classes.astype(np.int)

    data = data.iloc[:, :(n - 1)]
    data_new = data.iloc[:, [0, 2, 4, 10, 11, 12]]
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(data.iloc[:, [1, 3, 5, 6, 7, 8, 9, 13]])
    train = ohe.transform(data.iloc[:, [1, 3, 5, 6, 7, 8, 9, 13]])
    train = np.hstack([train, np.array(data_new)])
    if normalization:
        train = normalize(train)
    return train, classes
