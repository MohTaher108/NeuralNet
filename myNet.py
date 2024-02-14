# My custom neural network from scratch

import numpy as np
import matplotlib.pyplot as plt
import time

# Activation Functions
from resources import sigmoid, sigmoid_derivative
from resources import leaky_ReLU, leaky_ReLU_derivative
from resources import softmax, softmax_cross_entropy_derivative

# Loss Functions
from resources import MSE, MSE_derive
from resources import cross_entropy, cross_entropy_softmax_derive

# Helper Functions
from resources import shuffle_data, batch_split, multi_class_classification

test_debug = False



# Hidden layer contains a bunch of neurons
class HiddenLayer:
    def __init__(self, num_neurons, num_dimensions, activation, activation_derivative, dropout_rate):
        # weights 2-D array where each row is a neuron with a corresponding weight at each column
        self.weights = np.random.normal(size=(num_neurons, num_dimensions))
        # biases 1-D array where each element is a neuron's bias
        self.biases = np.random.normal(size=(num_neurons))
        # Activation functions
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.dropout_rate = dropout_rate
            

    def print_parameters(self):
        print("weights.shape = ", self.weights.shape, " and weights = \n", self.weights)
        print("biases.shape = ", self.biases.shape, " and biases = \n", self.biases)
    

    def count_parameters(self):
        return self.weights.size + self.biases.size


    def feed_forward(self, features):
        # Account for one data point edge case
        if features.ndim == 1:
            features = features[np.newaxis,:]

        result = self.activation(np.matmul(features, self.weights.T) + self.biases)

        # Dropout
        if not test_debug:
            dropout_size = np.floor(self.dropout_rate * result.shape[1]).astype(int)
            dropout_indices = np.random.choice(result.shape[1], size=dropout_size, replace=False)
            result[:,dropout_indices] = 0

        # Remove below code and properly scale dropout in backprop (included in todo.txt)
        # if not dropout_bool:
        #     if self.dropout_rate > 0:
        #         result *= self.dropout_rate

        return result



# Neural net which contains a list of hidden layers
class NeuralNet:
    # Initialize neural net with num_neurons_per_layer being a list of the input shape, layers' shapes, and output shape
    # def  __init__(self, num_neurons_per_layer, activation_funcs=[leaky_ReLU, sigmoid], activation_derivative_funcs=[leaky_ReLU_derivative, sigmoid_derivative], loss_func=MSE, loss_derivative_func=MSE_derive, dropout_rate=0.2):
    def __init__(self, num_neurons_per_layer, activation_funcs=[leaky_ReLU, softmax], activation_derivative_funcs=[leaky_ReLU_derivative, softmax_cross_entropy_derivative], loss_func=cross_entropy, loss_derivative_func=cross_entropy_softmax_derive, dropout_rate=0.2):
        self.num_layers = len(num_neurons_per_layer) - 1
        # Create layers
        self.layers = []
        for i in range(1, self.num_layers):
            self.layers.append(HiddenLayer(num_neurons_per_layer[i], num_neurons_per_layer[i-1], activation_funcs[0], activation_derivative_funcs[0], dropout_rate=dropout_rate))
        # Handle final layer's activation function
        self.layers.append(HiddenLayer(num_neurons_per_layer[self.num_layers], num_neurons_per_layer[self.num_layers-1], activation_funcs[1], activation_derivative_funcs[1], dropout_rate=dropout_rate))

        self.loss_func = loss_func
        self.loss_derivative_func = loss_derivative_func
        self.losses = []

        # Cache for testing (saves 2d array across epoch and layer)
        self.weight_updates = []
        self.bias_updates = []   
        self.neuron_partials = [] 

    def print_parameters(self, only_count=False):
        print("Number of parameters: ", self.count_parameters())
        if only_count: 
            return

        for i in range(self.num_layers):
            print("For layer ", i)
            self.layers[i].print_parameters()
            print()
    

    def count_parameters(self):
        count = 0
        for i in range(self.num_layers):
            count += self.layers[i].count_parameters()
        return count


    # Feed through every layer and return result (optionally return the full output of every layer or only last layer)
    def feed_forward(self, features, full_outputs=False, only_final_outputs=False):
        cur_output = features.copy()
        layers_outputs = [cur_output.copy()]
        for i in range(self.num_layers-1):
            cur_output = self.layers[i].feed_forward(cur_output)
            layers_outputs.append(cur_output.copy())
        # Finish last layer alone as it shouldn't have dropout
        cur_output = self.layers[self.num_layers-1].feed_forward(cur_output)
        layers_outputs.append(cur_output.copy())      

        if full_outputs:
            return layers_outputs
        if only_final_outputs:
            return cur_output
        else:
            return np.argmax(cur_output, axis=1)
    
    
    def train(self, features, labels, epochs=1000, batch_size=32, learning_rate=0.1, rel_tol=5e-4, loss_step=50):
        # If multi-label classification then switch labels to 2d array, where each row is an array with index label set to 1 and the rest to 0
        num_labels_possible = self.layers[self.num_layers - 1].biases.shape[0]
        if num_labels_possible > 1 and labels.ndim == 1:
            # aux_array = np.zeros((labels.shape[0], num_labels_possible))
            # z = np.arange(labels.shape[0])
            # aux_array[z, labels] = 1 # Index by 0,1,2,3 and labels to set correct values in array to 1
            # labels = aux_array
            labels = multi_class_classification(labels, num_labels_possible)

        start = time.time()
        self.gradient_descent(features, labels, epochs, batch_size, learning_rate, rel_tol, loss_step)
        print("Time elapsed during training: ", time.time() - start)


    def gradient_descent(self, features, labels, epochs, batch_size, learning_rate, rel_tol, loss_step):
        for epoch in range(epochs):
            print("\nepoch ", epoch)
            if test_debug:
                self.bias_updates.append([])
                self.weight_updates.append([])
                self.neuron_partials.append([])
            # Shuffle data
            if not test_debug:
                features, labels = shuffle_data(features, labels)
            # Back prop across all batches
            features_batches, labels_batches = batch_split(features, labels, batch_size)
            for features_batch, labels_batch in zip(features_batches, labels_batches):
                self.back_propagate(features_batch, labels_batch, learning_rate)
            # Check for convergence
            if epoch % loss_step == 0 and self.check_loss(features, labels, epoch, rel_tol):
                return
            
    
    def back_propagate(self, features, labels, learning_rate):
        layers_outputs = self.feed_forward(features, full_outputs=True) # layer 1's output is actually indexed by 1 (index 0 is original data)
        neuron_partials = self.loss_derivative_func(labels, layers_outputs[self.num_layers])
        # can simplify the below three lines to one line with something like reshape(-1, 1, np.shape[-1])
        if neuron_partials.ndim < 2:
            neuron_partials = neuron_partials.reshape((1, neuron_partials.shape[0]))
        neuron_partials = neuron_partials[:,np.newaxis,:]

        for layer in range(self.num_layers, 0, -1): # layer corresponds to which layer we're in indexed by 1
            if test_debug:
                self.neuron_partials[-1].append(neuron_partials)

            bias_updates = self.layers[layer-1].activation_derivative(layers_outputs[layer])[:, np.newaxis] * neuron_partials # check that broadcasting is correct on final layer (first layer in loop)
            bias_updates = np.mean(bias_updates, axis=1) # Average across every neuron path in the front layer
            weight_updates = bias_updates[:, :, np.newaxis] * layers_outputs[layer-1][:, np.newaxis,:]

            if layer > 1:
                product = self.layers[layer-1].weights.T[np.newaxis, :, :] * self.layers[layer-1].activation_derivative(layers_outputs[layer])[:, np.newaxis, :]
                neuron_partials = (np.mean(neuron_partials, axis=1, keepdims=True) * product)
                neuron_partials = neuron_partials.reshape(neuron_partials.shape[0], neuron_partials.shape[2], neuron_partials.shape[1])

            self.layers[layer-1].biases = self.layers[layer-1].biases - learning_rate * np.sum(bias_updates, axis=0) # play with sum vs mean
            self.layers[layer-1].weights = self.layers[layer-1].weights - learning_rate * np.sum(weight_updates, axis=0)

            if test_debug:
                self.weight_updates[-1].append(weight_updates)
                self.bias_updates[-1].append(bias_updates)


    # Returns true if loss change is less than rel_tol
    def check_loss(self, features, labels, epoch, rel_tol):
        prediction = self.feed_forward(features, only_final_outputs=True)
        loss = self.loss_func(labels, prediction)
        print("Epoch %3d loss: %.8f" % (epoch, loss))
        if len(self.losses) > 0:
            prev_loss = self.losses[len(self.losses)-1] # I think you can replace this with self.losses[-1]
            if np.abs(loss - prev_loss) / prev_loss < rel_tol:
                return True
        self.losses.append(loss)
        return False