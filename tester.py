# Tests myNet for a [2,2,2,1] network with 2 features coming in (pulled from files directory)

import numpy as np

def loadFromFile(fileName, arrayType):
    return np.loadtxt(fileName, dtype=arrayType)

label = loadFromFile('files/labels.txt', float)
weights_0 = loadFromFile('files/weights_0.txt', float)
weights_1 = loadFromFile('files/weights_1.txt', float)
weights_2 = loadFromFile('files/weights_2.txt', float)
weights_3 = np.array([loadFromFile('files/weights_3.txt', float)])
biases_0 = loadFromFile('files/biases_0.txt', float)
biases_1 = loadFromFile('files/biases_1.txt', float)
biases_2 = loadFromFile('files/biases_2.txt', float)
biases_3 = loadFromFile('files/biases_3.txt', float)
data = loadFromFile('files/layers_outputs_0.txt', float)
layers_outputs_0 = loadFromFile('files/layers_outputs_1.txt', float)
layers_outputs_1 = loadFromFile('files/layers_outputs_2.txt', float)
layers_outputs_2 = loadFromFile('files/layers_outputs_3.txt', float)
layers_outputs_3 = loadFromFile('files/layers_outputs_4.txt', float)

def print_params(label, weights, biases, data, layers_outputs):
    print("label = ", label)
    print("weights[0] = ", weights[0])
    print("biases[0] = ", biases[0])
    print("weights[1] = ", weights[1])
    print("biases[1] = ", biases[1])
    print("weights[2] = ", weights[2])
    print("biases[2] = ", biases[2])
    print("weights[3] = ", weights[3])
    print("biases[3] = ", biases[3])
    print("data = ", data)
    print("layers_outputs[0] = ", layers_outputs[0])
    print("layers_outputs[1] = ", layers_outputs[1])
    print("layers_outputs[2] = ", layers_outputs[2])
    print("layers_outputs[3] = ", layers_outputs[3])

# print_params(label, [weights_0, weights_1, weights_2, weights_3], [biases_0, biases_1, biases_2, biases_3], data, [layers_outputs_0, layers_outputs_1, layers_outputs_2, layers_outputs_3])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derive_sigmoid(x, do_sigmoid=False):
    sig = x.copy()
    if do_sigmoid: sig = sigmoid(sig)
    return sig * (1 - sig)

dL_dpred = -2 * (label - layers_outputs_3)

print("Layer 4")
neuron_partials = np.array([[dL_dpred]])
print("neuron_partials = \n", neuron_partials, "\n\n")
bias_changes = np.mean(neuron_partials * derive_sigmoid(layers_outputs_3), axis=0)
print("bias_changes = \n", bias_changes, "\n")
weight_changes = bias_changes * layers_outputs_2
print("weight_changes = \n", weight_changes, "\n")

print("Layer 3")
neuron_partials = (np.mean(neuron_partials, axis=0, keepdims=True) * (derive_sigmoid(layers_outputs_3) * weights_3).T).T
print("neuron_partials = \n", neuron_partials, "\n\n")
bias_changes = np.mean(neuron_partials * derive_sigmoid(layers_outputs_2), axis=0)
print("bias_changes = \n", bias_changes, "\n")
weight_changes = bias_changes[:, np.newaxis] * layers_outputs_1[np.newaxis, :]
print("weight_changes = \n", weight_changes, "\n")

print("Layer 2")
neuron_partials = (np.mean(neuron_partials, axis=0, keepdims=True) * (derive_sigmoid(layers_outputs_2) * weights_2).T).T
print("neuron_partials = \n", neuron_partials, "\n\n")
bias_changes = np.mean(neuron_partials * derive_sigmoid(layers_outputs_1), axis=0)
print("bias_changes = \n", bias_changes, "\n")
weight_changes = bias_changes[:, np.newaxis] * layers_outputs_0[np.newaxis, :]
print("weight_changes = \n", weight_changes, "\n")

print("Layer 1")
neuron_partials = (np.mean(neuron_partials, axis=0, keepdims=True) * (derive_sigmoid(layers_outputs_1) * weights_1).T).T
print("neuron_partials = \n", neuron_partials, "\n\n")
bias_changes = np.mean(neuron_partials * derive_sigmoid(layers_outputs_0), axis=0)
print("bias_changes = \n", bias_changes, "\n")
weight_changes = bias_changes[:, np.newaxis] * data[np.newaxis, :]
print("weight_changes = \n", weight_changes, "\n")

print("Layer 0")
neuron_partials = (np.mean(neuron_partials, axis=0, keepdims=True) * (derive_sigmoid(layers_outputs_0) * weights_0).T).T
print("neuron_partials = \n", neuron_partials, "\n\n")