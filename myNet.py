# My custom neural network from scratch

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derive_sigmoid(x, do_sigmoid=True):
    sig = x.copy()
    if do_sigmoid: sig = sigmoid(sig)
    return sig * (1 - sig)


def MSE_loss(truth, prediction):
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
    def feedForward(self, data):
        # Account for one data point edge case
        if data.ndim == 1:
            data = data[np.newaxis,:]
        result = np.sum(np.dot(data, self.weights.T), axis=0) + self.biases
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
    
    def print_parameters(self):
        for i in range(0, self.num_layers):
            print("For layer ", i)
            self.layers[i].print_parameters()
            print()

    # Feed through every layer and return result (optionally return the output of every layer)
    def feedForward(self, data, output_per_layer=False):
        cur_output = data.copy()
        layers_outputs = [cur_output.copy()]
        for i in range(self.num_layers):
            cur_output = self.layers[i].feedForward(cur_output)
            layers_outputs.append(cur_output.copy())

        if output_per_layer:
            return layers_outputs
        else:
            return cur_output[0]

    # good luck
    def train(self, data, learning_rate=0.1, epochs=1000, rel_tol=1e-5):
        features = data[:,:-1]
        labels = data[:,-1:].flatten()

        prev_loss = 0
        iter = 0 # deletee iter
        for epoch in range(epochs):
            for datapoint, label in zip(features, labels):
                layers_outputs = self.feedForward(datapoint, output_per_layer=True) # layer 1's output is actually indexed by 1 (index 0 is original data)
                dL_dpred = -2 * (label - layers_outputs[self.num_layers][0]) # index by zero since it returns array of one element
                neuron_partials = np.array([dL_dpred])

                # print("for iter ", iter) # deletee iter
                for layer in range(self.num_layers, 0, -1): #layer corresponds to which layer we're in indexed by 1
                    # print("neuron_partials = ", neuron_partials) # deletee
                    bias_changes = derive_sigmoid(layers_outputs[layer], do_sigmoid=False) * neuron_partials
                    # print("multiplying bias_changes to get weight_changes by \n", layers_outputs[layer-1])
                    weight_changes = np.matmul(bias_changes.T, layers_outputs[layer-1][np.newaxis,:])
                    
                    # print("for layer ", layer, " I'm updating weights with \n", weight_changes, "\nbiases with \n", bias_changes) # deletee
                    neuron_partials = neuron_partials * self.layers[layer-1].weights * derive_sigmoid(layers_outputs[layer], do_sigmoid=False)
                    self.layers[layer-1].biases = self.layers[layer-1].biases - learning_rate * bias_changes
                    self.layers[layer-1].weights = self.layers[layer-1].weights - learning_rate * weight_changes
                                        
                iter += 1 # deletee iter

            if epoch % 50 == 0:
                prediction = np.apply_along_axis(self.feedForward, 1, features)
                loss = MSE_loss(labels, prediction)
                print("Epoch %d loss: %.6f" % (epoch, loss))
                if epoch != 0 and np.abs(loss - prev_loss) / prev_loss < rel_tol:
                    return
                prev_loss = np.copy(loss)
                # self.losses.append(loss)


Alice = np.array([133, 65, 1], dtype='float')
Bob = np.array([160, 72, 0], dtype='float')
Charlie = np.array([152, 70, 0], dtype='float')
Diana = np.array([120, 60, 1], dtype='float')
data = np.array([Alice, Bob, Charlie, Diana])

data_averaged = np.mean(data[:,:-1], axis=0)
data[:,:-1] -= data_averaged
# print("data = ", data)

num_layers = 2
num_neurons_per_layer = [2,1]
num_dimensions = 2
neural_net = NeuralNet(num_layers, num_neurons_per_layer, num_dimensions)
neural_net.train(data, epochs=1000)
 
