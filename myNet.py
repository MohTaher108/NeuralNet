# My custom neural network from scratch

import numpy as np
import matplotlib.pyplot as plt
import time

# Activation Functions
from resources.activations import sigmoid, sigmoid_derivative
from resources.activations import Leaky_ReLU, Leaky_ReLU_derivative

# Loss Functions
from resources.losses import MSE 



# Hidden layer contains a bunch of neurons
class HiddenLayer:
    def __init__(self, num_neurons, num_dimensions, activation=sigmoid, activation_derivative=sigmoid_derivative):
        # weights 2-D array where each row is a neuron with a corresponding weight at each column
        self.weights = np.random.normal(size=(num_neurons, num_dimensions))
        # biases 1-D array where each element is a neuron's bias
        self.biases = np.random.normal(size=(num_neurons))
        # Activation functions
        self.activation = activation
        self.activation_derivative = activation_derivative

        self.old_weights = np.copy(self.weights)
        self.old_biases = np.copy(self.biases)
    

    def reset_weights(self):
        self.weights = self.old_weights
        self.biases = self.old_biases
    

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
    def __init__(self, num_layers, num_neurons_per_layer, num_dimensions, activation_funcs=[Leaky_ReLU, sigmoid], activation_derivative_funcs=[Leaky_ReLU_derivative, sigmoid_derivative]):
        self.losses = []
        self.num_layers = num_layers
        cur_layer = HiddenLayer(num_neurons_per_layer[0], num_dimensions)
        self.layers = [cur_layer]
        for i in range(1, num_layers-1):
            self.layers.append(HiddenLayer(num_neurons_per_layer[i], num_neurons_per_layer[i-1], activation_funcs[0], activation_derivative_funcs[0]))
        # Handle final activation
        self.layers.append(HiddenLayer(num_neurons_per_layer[num_layers-1], num_neurons_per_layer[num_layers-2], activation_funcs[1], activation_derivative_funcs[1]))
        

    def reset_weights(self):
        for i in range(self.num_layers):
            self.layers[i].reset_weights()
    

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
    
    
    def train(self, features, labels, GD_type="SGD", loss_func=MSE, epochs=1000, learning_rate=0.1, rel_tol=1e-4, loss_step=50):
        # If multi-label classification then switch labels to 2d array, where each row is an array with index label set to 1 and the rest to 0
        num_labels_possible = self.layers[self.num_layers - 1].biases.shape[0]
        if num_labels_possible > 1 and labels.ndim == 1:
            aux_array = np.zeros((labels.shape[0], num_labels_possible))
            z = np.arange(labels.shape[0])
            aux_array[z, labels] = 1 # Index by 0,1,2,3 and labels to set correct values in array to 1
            labels = aux_array
        
        start = time.time()
        if GD_type == "SGD":
            self.SGD(features, labels, loss_func=loss_func, epochs=epochs, learning_rate=learning_rate, rel_tol=rel_tol, loss_step=loss_step)
        elif GD_type == "GD":
            self.GD(features, labels, loss_func=loss_func, epochs=epochs, learning_rate=learning_rate, rel_tol=rel_tol, loss_step=loss_step)
        elif GD_type == "UGD":
            self.GD_unvectorized(features, labels, loss_func=loss_func, epochs=epochs, learning_rate=learning_rate, rel_tol=rel_tol, loss_step=loss_step)
        else:
            print("Error: invalid GD training type")
        print("Time elapsed during training: ", time.time() - start)


    def GD(self, features, labels, loss_func=MSE, epochs=1000, learning_rate=0.1, rel_tol=1e-4, loss_step=50):
        for epoch in range(epochs):
            layers_outputs = self.feed_forward(features, output_per_layer=True) # layer 1's output is actually indexed by 1 (index 0 is original data)
            dL_dpred = loss_func(labels, layers_outputs[self.num_layers], derive=True)
            neuron_partials = dL_dpred
            if neuron_partials.ndim < 2:
                neuron_partials = neuron_partials.reshape((1, neuron_partials.shape[0]))
            neuron_partials = neuron_partials[:,np.newaxis,:]
            print("neuron_partials[0] = ", neuron_partials[0])

            for layer in range(self.num_layers, 0, -1): # layer corresponds to which layer we're in indexed by 1
                bias_changes = self.layers[layer-1].activation_derivative(layers_outputs[layer])[:, np.newaxis] * neuron_partials
                bias_changes = np.mean(bias_changes, axis=1) # Average across every neuron path in the front layer
                weight_changes = bias_changes[:, :, np.newaxis] * layers_outputs[layer-1][:, np.newaxis,:]
                
                if layer > 1:
                    product = self.layers[layer-1].weights.T[np.newaxis, :, :] * self.layers[layer-1].activation_derivative(layers_outputs[layer])[:, np.newaxis, :]
                    print("product[0] = ", product[0])
                    print("meaned neuron_partials = ", np.mean(neuron_partials, axis=1, keepdims=True)[0])
                    neuron_partials = (np.mean(neuron_partials, axis=1, keepdims=True) * product)
                    neuron_partials = neuron_partials.reshape(neuron_partials.shape[0], neuron_partials.shape[2], neuron_partials.shape[1])
                    print("neuron_partials[0] = ", neuron_partials[0])

                fileName = 'files/bias_changes_' + str(layer) + '.txt'
                saveToFile(fileName, bias_changes[0], '%f')
                fileName = 'files/weight_changes_' + str(layer) + '.txt'
                saveToFile(fileName, weight_changes[0], '%f')

                self.layers[layer-1].biases = self.layers[layer-1].biases - learning_rate * np.sum(bias_changes, axis=0) # play with sum vs mean
                self.layers[layer-1].weights = self.layers[layer-1].weights - learning_rate * np.sum(weight_changes, axis=0)

            if epoch % loss_step == 0:
                self.check_loss(features, labels, epoch, loss_func=loss_func, rel_tol=rel_tol)


    def GD_unvectorized2(self, features, labels, loss_func=MSE, epochs=1000, learning_rate=0.1, rel_tol=1e-4, loss_step=50):
        for epoch in range(epochs):
            features, labels = shuffle_data(features, labels)
            layers_outputs = self.feed_forward(features, output_per_layer=True)
            dL_dpred = loss_func(labels, layers_outputs[self.num_layers], derive=True)
            neuron_partials = dL_dpred
            if neuron_partials.ndim < 2:
                neuron_partials = neuron_partials.reshape((1, neuron_partials.shape[0]))
            neuron_partials = neuron_partials[:,np.newaxis,:]

            bias_changes_list_print = []
            weight_changes_list_print = []
            for layer in range(self.num_layers, 0, -1):
                new_neuron_partials = []
                bias_changes_list = []
                weight_changes_list = []
                for i in range(len(features)):
                    bias_changes = self.layers[layer-1].activation_derivative(layers_outputs[layer][i]) * neuron_partials[i]
                    bias_changes = np.mean(bias_changes, axis=0) # Average across every neuron path in the front layer
                    weight_changes = bias_changes[:, np.newaxis] * layers_outputs[layer-1][i][np.newaxis,:]
                    
                    if layer > 1:
                        product = self.layers[layer-1].weights.T * self.layers[layer-1].activation_derivative(layers_outputs[layer][i])
                        new_neuron_partials.append((np.mean(neuron_partials[i], axis=0, keepdims=True) * product).T)  
                    bias_changes_list.append(bias_changes)
                    weight_changes_list.append(weight_changes)

                    if i == 0:
                        bias_changes_list_print.append(np.copy(bias_changes))
                        weight_changes_list_print.append(np.copy(weight_changes))

                neuron_partials = np.array(new_neuron_partials)
                self.layers[layer-1].biases = self.layers[layer-1].biases - learning_rate * np.sum(np.array(bias_changes_list), axis=0)
                self.layers[layer-1].weights = self.layers[layer-1].weights - learning_rate * np.sum(np.array(weight_changes_list), axis=0)
            
            # for j in range(self.num_layers):
            #     index = self.num_layers - j
            #     fileName = 'files/UGD_bias_changes_list_' + str(index) + '.txt'
            #     saveToFile(fileName, bias_changes_list_print[j], '%f')
            #     fileName = 'files/UGD_weight_changes_list_' + str(index) + '.txt'
            #     saveToFile(fileName, weight_changes_list_print[j], '%f')
            
            if epoch % loss_step == 0:
                self.check_loss(features, labels, epoch, loss_func=loss_func, rel_tol=rel_tol)

    
    def GD_unvectorized(self, features, labels, loss_func=MSE, epochs=1000, learning_rate=0.1, rel_tol=1e-4, loss_step=50):
        for epoch in range(epochs):
            bias_changes_list = [[] for i in range(self.num_layers)]
            weight_changes_list = [[] for i in range(self.num_layers)]
            features, labels = shuffle_data(features, labels)
            for datapoint, label in zip(features, labels):
                layers_outputs = self.feed_forward(datapoint, output_per_layer=True) # layer 1's output is actually indexed by 1 (index 0 is original data)
                dL_dpred = loss_func(label, layers_outputs[self.num_layers], derive=True) 
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
            if epoch % loss_step == 0:
                self.check_loss(features, labels, epoch, loss_func=loss_func, rel_tol=rel_tol)


    def SGD(self, features, labels, loss_func=MSE, epochs=1000, learning_rate=0.1, rel_tol=1e-4, loss_step=50):
        for epoch in range(epochs):
            features, labels = shuffle_data(features, labels)
            for datapoint, label in zip(features, labels):
                self.back_propagate(datapoint, label, loss_func=loss_func, learning_rate=learning_rate)
            if epoch % loss_step == 0:
                self.check_loss(features, labels, epoch, loss_func=loss_func, rel_tol=rel_tol)
    

    def back_propagate(self, features, labels, loss_func=MSE, learning_rate=0.1):
        layers_outputs = self.feed_forward(features, output_per_layer=True) # layer 1's output is actually indexed by 1 (index 0 is original data)
        dL_dpred = loss_func(labels, layers_outputs[self.num_layers], derive=True)
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
    

    def check_loss(self, features, labels, epoch, loss_func=MSE, rel_tol=1e-4):
        prediction = np.apply_along_axis(self.feed_forward, 1, features, full_final_output=True)
        loss = loss_func(labels, prediction)
        print("Epoch %3d loss: %.8f" % (epoch, loss))
        if len(self.losses) > 0:
            prev_loss = self.losses[len(self.losses)-1]
            if np.abs(loss - prev_loss) / prev_loss < rel_tol:
                return
        self.losses.append(loss)
    


def shuffle_data(features, labels):
    perm = np.random.permutation(features.shape[0])
    return features[perm], labels[perm]


def saveToFile(fileName, array, arrayTypeIdentifier):
    np.savetxt(fileName, array, fmt=arrayTypeIdentifier)