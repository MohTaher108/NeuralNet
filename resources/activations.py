# Activation functions such as sigmoid, softmax, ReLU, and Leaky ReLU

import numpy as np

alpha = 0.1
def leaky_ReLU(input):
    res = input.copy()
    res[res < 0] *= alpha
    return res

def leaky_ReLU_derivative(input):
    res = input.copy()
    res[res < 0] = alpha
    res[res > 0] = 1
    return res



def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def sigmoid_derivative(input):
    return input * (1 - input)



# Using softmax activation currently requires using cross entropy loss for backpropagation derivation at final layer
def softmax(input):
    input = input - np.max(input, axis=-1, keepdims=True) # Prevent an exploding value
    exponent = np.exp(input)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)

def softmax_cross_entropy_derivative(input):
    return np.ones(input.shape)