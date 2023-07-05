# Activation functions such as sigmoid, softmainput, ReLU, and Leaky ReLU

import numpy as np


def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def sigmoid_derivative(input): # Expected sigmoid values
    return input * (1 - input)



alpha = 0.1
def Leaky_ReLU(input):
    res = input.copy()
    res[res < 0] *= alpha
    return res

def Leaky_ReLU_derivative(input):
    res = input.copy()
    res[res < 0] = alpha
    res[res > 0] = 1
    return res



# def softmax(input):
#     return 