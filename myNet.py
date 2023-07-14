# My custom neural network from scratch

import numpy as np
import matplotlib.pyplot as plt
import time

# Activation Functions
from resources.activations import sigmoid, sigmoid_derivative
from resources.activations import leaky_ReLU, leaky_ReLU_derivative
from resources.activations import softmax, softmax_cross_entropy_derivative

# Loss Functions
from resources.losses import MSE, MSE_derive
from resources.losses import cross_entropy, cross_entropy_softmax_derive

# Hidden layer contains a bunch of neurons
class HiddenLayer:
    def __init__(self, num_neurons, num_dimensions, activation, activation_derivative):
        # weights 2-D array where each row is a neuron with a corresponding weight at each column
        self.weights = np.random.normal(size=(num_neurons, num_dimensions))
        # biases 1-D array where each element is a neuron's bias
        self.biases = np.random.normal(size=(num_neurons))
        # Activation functions
        self.activation = activation
        self.activation_derivative = activation_derivative

        self.old_weights = np.copy(self.weights)
        self.old_biases = np.copy(self.biases)

        self.num_neurons = num_neurons
        self.num_dimensions = num_dimensions
    

    def reset_weights(self):
        self.weights = self.old_weights
        self.biases = self.old_biases
    

    def init_weights(self):
        self.weights = np.random.normal(size=(self.num_neurons, self.num_dimensions))
        self.biases = np.random.normal(size=(self.num_neurons))


    def print_parameters(self):
        print("weights.shape = ", self.weights.shape, " and weights = \n", self.weights)
        print("biases.shape = ", self.biases.shape, " and biases = \n", self.biases)
    

    def count_parameters(self):
        return self.weights.size + self.biases.size


    def feed_forward(self, features):
        # Account for one data point edge case
        if features.ndim == 1:
            features = features[np.newaxis,:]
            
        result = np.matmul(features, self.weights.T) + self.biases
        if result.shape[0] == 1: # Remove unnecessary 2nd dimension
            result = result[0]

        return self.activation(result)



# Neural net which contains a list of hidden layers
class NeuralNet:
    # Initialize neural net with num_neurons_per_layer being a list of the input shape, layers' shapes, and output shape
    def __init__(self, num_neurons_per_layer, activation_funcs=[leaky_ReLU, softmax], activation_derivative_funcs=[leaky_ReLU_derivative, softmax_cross_entropy_derivative], loss_func=cross_entropy, loss_derivative_func=cross_entropy_softmax_derive):
    # def __init__(self, num_neurons_per_layer, activation_funcs=[leaky_ReLU, sigmoid], activation_derivative_funcs=[leaky_ReLU_derivative, sigmoid_derivative], loss_func=MSE, loss_derivative_func=MSE_derive):
        self.num_layers = len(num_neurons_per_layer) - 1
        # Create layers
        self.layers = []
        for i in range(1, self.num_layers):
            self.layers.append(HiddenLayer(num_neurons_per_layer[i], num_neurons_per_layer[i-1], activation_funcs[0], activation_derivative_funcs[0]))
        # Handle final layer's activation function
        self.layers.append(HiddenLayer(num_neurons_per_layer[self.num_layers], num_neurons_per_layer[self.num_layers-1], activation_funcs[1], activation_derivative_funcs[1]))

        self.loss_func = loss_func
        self.loss_derivative_func = loss_derivative_func
        self.losses = []
        

    def reset_weights(self):
        for i in range(self.num_layers):
            self.layers[i].reset_weights()
    

    def init_weights(self):
        for i in range(self.num_layers):
            self.layers[i].init_weights()


    def print_parameters(self):
        print("Number of parameters: ", self.count_parameters())
        for i in range(self.num_layers):
            print("For layer ", i)
            self.layers[i].print_parameters()
            print()
    

    def count_parameters(self):
        count = 0
        for i in range(self.num_layers):
            count += self.layers[i].count_parameters()
        return count


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
    
    
    def train(self, features, labels, GD_type="SGD", epochs=1000, learning_rate=0.1, rel_tol=1e-4, loss_step=50):
        # If multi-label classification then switch labels to 2d array, where each row is an array with index label set to 1 and the rest to 0
        num_labels_possible = self.layers[self.num_layers - 1].biases.shape[0]
        if num_labels_possible > 1 and labels.ndim == 1:
            aux_array = np.zeros((labels.shape[0], num_labels_possible))
            z = np.arange(labels.shape[0])
            aux_array[z, labels] = 1 # Index by 0,1,2,3 and labels to set correct values in array to 1
            labels = aux_array
        
        start = time.time()
        if GD_type == "SGD":
            self.SGD(features, labels, epochs, learning_rate, rel_tol, loss_step)
        elif GD_type == "GD":
            self.GD(features, labels, epochs, learning_rate, rel_tol, loss_step)
        else:
            print("Error: invalid GD training type")
        print("Time elapsed during training: ", time.time() - start)


    def BGD(self, features, labels, epochs, learning_rate, rel_tol, loss_step, batch_size=32):
        for epoch in range(epochs):
            layers_outputs = self.feed_forward(features, output_per_layer=True) # layer 1's output is actually indexed by 1 (index 0 is original data)
            dL_dpred = self.loss_derivative_func(labels, layers_outputs[self.num_layers])
            neuron_partials = dL_dpred
            if neuron_partials.ndim < 2:
                neuron_partials = neuron_partials.reshape((1, neuron_partials.shape[0]))
            neuron_partials = neuron_partials[:,np.newaxis,:]

            for layer in range(self.num_layers, 0, -1): # layer corresponds to which layer we're in indexed by 1
                # features_batches, labels_batches = self.batchSplit()
                # for features_batch, labels_batch in zip(features_batches, labels_batches):
                    bias_changes = self.layers[layer-1].activation_derivative(layers_outputs[layer])[:, np.newaxis] * neuron_partials
                    bias_changes = np.mean(bias_changes, axis=1) # Average across every neuron path in the front layer
                    weight_changes = bias_changes[:, :, np.newaxis] * layers_outputs[layer-1][:, np.newaxis,:]
                    
                    if layer > 1:
                        product = self.layers[layer-1].weights.T[np.newaxis, :, :] * self.layers[layer-1].activation_derivative(layers_outputs[layer])[:, np.newaxis, :]
                        neuron_partials = (np.mean(neuron_partials, axis=1, keepdims=True) * product)
                        neuron_partials = neuron_partials.reshape(neuron_partials.shape[0], neuron_partials.shape[2], neuron_partials.shape[1])

                    self.layers[layer-1].biases = self.layers[layer-1].biases - learning_rate * np.sum(bias_changes, axis=0) # play with sum vs mean
                    self.layers[layer-1].weights = self.layers[layer-1].weights - learning_rate * np.sum(weight_changes, axis=0)

            if epoch % loss_step == 0 and self.check_loss(features, labels, epoch, rel_tol):
                return
            
    
    def batchSplit():
        print()


    def GD(self, features, labels, epochs, learning_rate, rel_tol, loss_step):
        for epoch in range(epochs):
            layers_outputs = self.feed_forward(features, output_per_layer=True) # layer 1's output is actually indexed by 1 (index 0 is original data)
            dL_dpred = self.loss_derivative_func(labels, layers_outputs[self.num_layers])
            neuron_partials = dL_dpred
            if neuron_partials.ndim < 2:
                neuron_partials = neuron_partials.reshape((1, neuron_partials.shape[0]))
            neuron_partials = neuron_partials[:,np.newaxis,:]

            for layer in range(self.num_layers, 0, -1): # layer corresponds to which layer we're in indexed by 1
                bias_changes = self.layers[layer-1].activation_derivative(layers_outputs[layer])[:, np.newaxis] * neuron_partials
                bias_changes = np.mean(bias_changes, axis=1) # Average across every neuron path in the front layer
                weight_changes = bias_changes[:, :, np.newaxis] * layers_outputs[layer-1][:, np.newaxis,:]
                
                if layer > 1:
                    product = self.layers[layer-1].weights.T[np.newaxis, :, :] * self.layers[layer-1].activation_derivative(layers_outputs[layer])[:, np.newaxis, :]
                    neuron_partials = (np.mean(neuron_partials, axis=1, keepdims=True) * product)
                    neuron_partials = neuron_partials.reshape(neuron_partials.shape[0], neuron_partials.shape[2], neuron_partials.shape[1])

                self.layers[layer-1].biases = self.layers[layer-1].biases - learning_rate * np.sum(bias_changes, axis=0) # play with sum vs mean
                self.layers[layer-1].weights = self.layers[layer-1].weights - learning_rate * np.sum(weight_changes, axis=0)

            if epoch % loss_step == 0 and self.check_loss(features, labels, epoch, rel_tol):
                return
            
    
    def GD_unvectorized(self, features, labels, epochs, learning_rate, rel_tol, loss_step):
        for epoch in range(epochs):
            bias_changes_list = [[] for i in range(self.num_layers)]
            weight_changes_list = [[] for i in range(self.num_layers)]
            features, labels = shuffle_data(features, labels)
            for datapoint, label in zip(features, labels):
                layers_outputs = self.feed_forward(datapoint, output_per_layer=True) # layer 1's output is actually indexed by 1 (index 0 is original data)
                dL_dpred = self.loss_derivative_func(label, layers_outputs[self.num_layers]) 
                neuron_partials = dL_dpred
                if neuron_partials.ndim < 2:
                    neuron_partials = neuron_partials.reshape((1, neuron_partials.shape[0]))
                
                i = -1
                for layer in range(self.num_layers, 0, -1): # layer corresponds to which layer we're in indexed by 1
                    i += 1
                    bias_changes = self.layers[layer-1].activation_derivative(layers_outputs[layer]) * neuron_partials
                    bias_changes = np.mean(bias_changes, axis=0) # Average across every neuron path in the front layer
                    weight_changes = bias_changes[:, np.newaxis] * layers_outputs[layer-1][np.newaxis,:]
                    
                    if layer > 1:
                        product = self.layers[layer-1].weights.T * self.layers[layer-1].activation_derivative(layers_outputs[layer])
                        neuron_partials = (np.mean(neuron_partials, axis=0, keepdims=True) * product).T
                    bias_changes_list[i].append(bias_changes)
                    weight_changes_list[i].append(weight_changes)
            
            for layer in range(self.num_layers, 0, -1):
                # print("bias_changes at layer ", layer, " = ", np.array(bias_changes_list[self.num_layers - layer]).shape)
                self.layers[layer-1].biases = self.layers[layer-1].biases - learning_rate * np.sum(np.array(bias_changes_list[self.num_layers - layer]), axis=0)
                self.layers[layer-1].weights = self.layers[layer-1].weights - learning_rate * np.sum(np.array(weight_changes_list[self.num_layers - layer]), axis=0)
            if epoch % loss_step == 0 and self.check_loss(features, labels, epoch, rel_tol):
                return


    def SGD(self, features, labels, epochs, learning_rate, rel_tol, loss_step):
        for epoch in range(epochs):
            features, labels = shuffle_data(features, labels)
            for datapoint, label in zip(features, labels):
                self.back_propagate(datapoint, label, learning_rate)
            if epoch % loss_step == 0 and self.check_loss(features, labels, epoch, rel_tol):
                return
    

    def back_propagate(self, features, labels, learning_rate):
        layers_outputs = self.feed_forward(features, output_per_layer=True) # layer 1's output is actually indexed by 1 (index 0 is original data)
        dL_dpred = self.loss_derivative_func(labels, layers_outputs[self.num_layers])
        neuron_partials = dL_dpred
        if neuron_partials.ndim < 2:
            neuron_partials = neuron_partials.reshape((1, neuron_partials.shape[0]))
        
        for layer in range(self.num_layers, 0, -1): # layer corresponds to which layer we're in indexed by 1 
            bias_changes = self.layers[layer-1].activation_derivative(layers_outputs[layer]) * neuron_partials
            bias_changes = np.mean(bias_changes, axis=0) # Average across every neuron path in the front layer
            weight_changes = bias_changes[:, np.newaxis] * layers_outputs[layer-1][np.newaxis,:]
            
            if layer > 1:
                product = self.layers[layer-1].weights.T * self.layers[layer-1].activation_derivative(layers_outputs[layer])
                neuron_partials = (np.mean(neuron_partials, axis=0, keepdims=True) * product).T
            self.layers[layer-1].biases = self.layers[layer-1].biases - learning_rate * bias_changes
            self.layers[layer-1].weights = self.layers[layer-1].weights - learning_rate * weight_changes
    

    # Returns true if loss change is less than rel_tol
    def check_loss(self, features, labels, epoch, rel_tol):
        prediction = np.apply_along_axis(self.feed_forward, 1, features, full_final_output=True)
        loss = self.loss_func(labels, prediction)
        print("Epoch %3d loss: %.8f" % (epoch, loss))
        if len(self.losses) > 0:
            prev_loss = self.losses[len(self.losses)-1]
            if np.abs(loss - prev_loss) / prev_loss < rel_tol:
                return True
        self.losses.append(loss)
        return False
    


def shuffle_data(features, labels):
    perm = np.random.permutation(features.shape[0])
    return features[perm], labels[perm]