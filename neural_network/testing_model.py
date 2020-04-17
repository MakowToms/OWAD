import numpy as np


def MSE(x, y):
    return np.mean((x - y) ** 2)


def accuracy(result, y_numeric):
    return np.sum(y_numeric.transpose()==result)/y_numeric.shape[0]


# function create change number lists to have the same length
# it fills needed values with the same value as last from list
def add_the_same_value_to_the_same_length(numbers_list_list):
    max_length = 0
    for i in range(len(numbers_list_list)):
        if len(numbers_list_list[i]) > max_length:
            max_length = len(numbers_list_list[i])
    for i in range(len(numbers_list_list)):
        value = numbers_list_list[i][-1]
        for j in range(len(numbers_list_list[i]), max_length):
            numbers_list_list[i].append(value)
    return numbers_list_list

