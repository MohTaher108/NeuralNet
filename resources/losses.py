# Loss functions such as MSE and Cross-Entropy

import numpy as np


def MSE(truth, prediction):
    return np.mean((truth - prediction) ** 2)

def MSE_derive(truth, prediction):
    return -2 * (truth - prediction)



def cross_entropy(truth, prediction):
    return -1 * np.mean(np.sum(truth * np.log(prediction + 1e-15), axis=-1))

def cross_entropy_softmax_derive(truth, prediction):
    return (prediction - truth)