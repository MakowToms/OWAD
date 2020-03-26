from numpy import genfromtxt
from neural_network.activations import *
import pandas as pd


def normalize(X):
    return (X-np.mean(X, 0))/np.std(X, 0)


# only one case to use this function - data easy (classification)
# because numpy don't support strings
def load_easy_data(normalization=True):
    train = load_data_with_true_false('mio1/' + 'classification' + '/' + 'easy' + '-training.csv')
    test = load_data_with_true_false('mio1/' + 'classification' + '/' + 'easy' + '-test.csv')
    if normalization:
        train[:, :-1] = normalize(train[:, :-1])
        test[:, :-1] = normalize(test[:, :-1])
    return train, test


# only one case to use this function - data easy (classification)
# because numpy don't support strings
def load_data_with_true_false(file_name):
    df = pd.read_csv(file_name)
    the_column_should_not_be_in_data = df.iloc[:, 2]
    the_column_should_not_be_in_data[the_column_should_not_be_in_data] = 1
    the_column_should_not_be_in_data[the_column_should_not_be_in_data==False] = 0
    df.iloc[:, 2] = the_column_should_not_be_in_data
    return np.array(df)


# load typical data as numpy arrays
def load_data(file_name, folder='regression', normalization=True, classification=False):
    train = genfromtxt('mio1/' + folder + '/' + file_name + '-training.csv', delimiter=',')
    if not classification:
        train = train[1:, 1:]
    else:
        train = train[1:, :]
    test = genfromtxt('mio1/' + folder + '/' + file_name + '-test.csv', delimiter=',')
    if not classification:
        test = test[1:, 1:]
    else:
        train = train[1:, :]
    if normalization:
        if not classification:
            train = normalize(train)
            test = normalize(test)
        else:
            train[:, :-1] = normalize(train[:, :-1])
            test[:, :-1] = normalize(test[:, :-1])
    return train, test


def one_hot_encode(y):
    uniques = np.unique(y)
    result = np.zeros([y.shape[0], uniques.shape[0]])
    for index, unique in enumerate(uniques):
        index_array = y==uniques[index]
        index_array = index_array.reshape(y.shape[0])
        result[index_array, index] = 1
    return result
