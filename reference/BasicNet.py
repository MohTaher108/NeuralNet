# Basic neural network from https://victorzhou.com/blog/intro-to-neural-networks/

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def MSE_loss(truth, prediction):
    return np.mean((truth - prediction) ** 2)

# y_true = np.array([1,0,0,1])
# y_pred = np.array([0,0,1,1])
# print(MSE_loss(y_true, y_pred))

# Neuron class contains the weights and bias for every neuron
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedForward(self, data):
        return sigmoid(np.sum(np.dot(self.weights, data)) + self.bias)

# Hidden layer contains a list of neurons
class HiddenLayer:
    def __init__(self, weights_array, bias_array):
        self.neurons = []
        for i in range(len(weights_array)):
            self.neurons.append(Neuron(weights_array[i], bias_array[i]))
    
    def feedForward(self, data):
        result = []
        for i in range(len(self.neurons)):
            result.append(self.neurons[i].feedForward(data))
        return result

# Neural net which contains a list of hidden layers
class NeuralNet:
    def __init__(self, weights_arrays, bias_arrays):
        self.layers = []
        for i in range(len(weights_arrays)):
            cur_layer = HiddenLayer(weights_arrays[i], bias_arrays[i])
            self.layers.append(cur_layer)
    
    def feedForward(self, data):
        cur_data = data.copy()
        for i in range(len(self.layers)):
            cur_data = self.layers[i].feedForward(cur_data)
        return cur_data[0]

weights = np.array([0,1])
bias = 0
weights_arrays = [[weights, weights], [weights]]
bias_arrays = [[bias, bias], [bias]]
cur_neural_net = NeuralNet(weights_arrays, bias_arrays)
print(cur_neural_net.feedForward(np.array([2,3])))