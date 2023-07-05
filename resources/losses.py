# Loss functions such as MSE and Cross-Entropy

import numpy as np

def MSE(truth, prediction, derive=False):
    if derive:
        return MSE_derive(truth, prediction)

    if truth.shape != prediction.shape:
        print("MSE shape mismatch!")  
    return np.mean((truth - prediction) ** 2)

def MSE_derive(truth, prediction):
    return -2 * (truth - prediction)



# def crossEntropy(input):
#     return 