import numpy as np

# Activation Functions
from resources import sigmoid, sigmoid_derivative
from resources import leaky_ReLU, leaky_ReLU_derivative
from resources import softmax, softmax_cross_entropy_derivative

# Loss Functions
from resources import MSE, MSE_derive
from resources import cross_entropy, cross_entropy_softmax_derive

# Helper Functions
from resources import shuffle_data, batch_split, multi_class_classification



class dummyNet():
    def __init__(self, num_neurons_per_layer, activation_funcs=[leaky_ReLU, softmax], activation_derivative_funcs=[leaky_ReLU_derivative, softmax_cross_entropy_derivative], loss_func=cross_entropy, loss_derivative_func=cross_entropy_softmax_derive):
        self.weights = []
        self.biases = []
        self.num_layers = len(num_neurons_per_layer) - 1
        for i in range(1, self.num_layers + 1):
            self.weights.append(np.random.normal(size=(num_neurons_per_layer[i], num_neurons_per_layer[i-1])))
            self.biases.append(np.random.normal(size=(num_neurons_per_layer[i])))

        self.hid_act_func = activation_funcs[0]
        self.out_act_func = activation_funcs[1]
        self.hid_act_der_func = activation_derivative_funcs[0]
        self.out_act_der_func = activation_derivative_funcs[1]

        self.loss_func = loss_func
        self.loss_der_func = loss_derivative_func


    def feed_forward(self, features, full_outputs=False, only_final_outputs=False):
        if features.ndim == 1:
            features = features[np.newaxis,:]
        
        cur_output = features.copy()
        layers_outputs = [cur_output.copy()]
        for i in range(self.num_layers):
            cur_output = self.activation(np.matmul(cur_output, self.weights[i].T) + self.biases[i])
            layers_outputs.append(cur_output.copy()) 

        if full_outputs:
            return layers_outputs
        if only_final_outputs:
            return cur_output
        else:
            return np.argmax(cur_output, axis=1)