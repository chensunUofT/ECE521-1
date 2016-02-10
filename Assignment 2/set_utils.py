from random import shuffle
import numpy as np

def make_sets(input, classes, train=1500, validation=1000):

    # shuffle(source)

    x_train = input[:train]
    t_train = classes[:train]

    x_validation = input[train:validation+train]
    t_validation = classes[train:validation+train]

    x_test = input[validation+train:]
    t_test = classes[validation+train:]

    return x_train, t_train, x_validation, t_validation, x_test, t_test

def to_one_hot(input):
    num_classes = max(input)+1

    one_hot = np.zeros([len(input), num_classes])

    for i, hot in enumerate(input):
        one_hot[i, hot] = 1

    return one_hot