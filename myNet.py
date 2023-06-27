# My custom neural network from scratch

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derive_sigmoid(x, do_sigmoid=True):
    sig = x.copy()
    if do_sigmoid: sig = sigmoid(sig)
    return sig * (1 - sig)

def brier_loss(truth, prediction):
    if truth.shape != prediction.shape:
        print("brier_loss shape mismatch!")
    return np.mean((truth - prediction) ** 2)


# Hidden layer contains a bunch of neurons
class HiddenLayer:
    def __init__(self, num_neurons, num_dimensions):
        # weights 2-D array where each row is a neuron with a corresponding weight at each column
        self.weights = np.random.normal(size=(num_neurons, num_dimensions))
        # biases 1-D array where each element is a neuron's bias
        self.biases = np.random.normal(size=(num_neurons))
    
    def print_parameters(self, biases_size=True):
        print("weights.shape = ", self.weights.shape, " and weights = \n", self.weights)
        print("biases.shape = ", self.biases.shape, " and biases = \n", self.biases)
    
    # Feed through layer with sigmoid activation
    def feed_forward(self, features):
        # Account for one data point edge case
        if features.ndim == 1:
            features = features[np.newaxis,:]
            
        result = np.matmul(features, self.weights.T) + self.biases
        if result.shape[0] == 1: # Remove unnecessary 2nd dimension
            result = result[0]

        return sigmoid(result)

# Neural net which contains a list of hidden layers
class NeuralNet:
    def __init__(self, num_layers, num_neurons_per_layer, num_dimensions):
        self.num_layers = num_layers
        cur_layer = HiddenLayer(num_neurons_per_layer[0], num_dimensions)
        self.layers = [cur_layer]
        for i in range(1, num_layers):
            cur_layer = HiddenLayer(num_neurons_per_layer[i], num_neurons_per_layer[i-1])
            self.layers.append(cur_layer)
        self.losses = []
    
    def print_parameters(self):
        for i in range(0, self.num_layers):
            print("For layer ", i)
            self.layers[i].print_parameters()
            print()

    # Feed through every layer and return result (optionally return the output of every layer)
    def feed_forward(self, features, output_per_layer=False, full_final_output=False):
        cur_output = features.copy()
        layers_outputs = [cur_output.copy()]
        for i in range(self.num_layers):
            cur_output = self.layers[i].feed_forward(cur_output)
            layers_outputs.append(cur_output.copy())            

        if output_per_layer:
            return layers_outputs
        if full_final_output:
            return cur_output
        else:
            return np.argmax(cur_output, axis=1)

    # Train the network using SGD
    def train(self, features, labels, learning_rate=0.1, epochs=1000, rel_tol=1e-5):
        # If multi-label classification then switch labels to 2d array, where each row is an array with index label set to 1 and the rest to 0
        num_labels_possible = self.layers[self.num_layers - 1].biases.shape[0]
        if num_labels_possible > 1 and labels.ndim == 1:
            aux_array = np.zeros((labels.shape[0], num_labels_possible))
            z = np.arange(labels.shape[0])
            aux_array[z, labels] = 1 # Index by 0,1,2,3 and labels to set correct values in array to 1
            labels = aux_array
        
        prev_loss = 0
        for epoch in range(epochs):
            i = 0
            for datapoint, label in zip(features, labels):
                i += 1
                layers_outputs = self.feed_forward(datapoint, output_per_layer=True) # layer 1's output is actually indexed by 1 (index 0 is original data)
                dL_dpred = -2 * (label - layers_outputs[self.num_layers])
                neuron_partials = np.array([dL_dpred])

                for layer in range(self.num_layers, 0, -1): # layer corresponds to which layer we're in indexed by 1
                    bias_changes = derive_sigmoid(layers_outputs[layer], do_sigmoid=False) * neuron_partials
                    bias_changes = np.mean(bias_changes, axis=0) # Average across every neuron path in the front layer
                    weight_changes = bias_changes[:, np.newaxis] * layers_outputs[layer-1][np.newaxis,:]
                    
                    if layer > 1:
                        product = self.layers[layer-1].weights.T * derive_sigmoid(layers_outputs[layer], do_sigmoid=False)
                        neuron_partials = (np.mean(neuron_partials, axis=0, keepdims=True) * product).T
                    self.layers[layer-1].biases = self.layers[layer-1].biases - learning_rate * bias_changes
                    self.layers[layer-1].weights = self.layers[layer-1].weights - learning_rate * weight_changes

            if epoch % 50 == 0:
                prediction = np.apply_along_axis(self.feed_forward, 1, features, full_final_output=True)
                loss = brier_loss(labels, prediction)
                print("Epoch %d loss: %.6f" % (epoch, loss))
                if epoch != 0 and np.abs(loss - prev_loss) / prev_loss < rel_tol:
                    print("loss ratio = ", np.abs(loss - prev_loss) / prev_loss)
                    return
                prev_loss = np.copy(loss)
                self.losses.append(loss)