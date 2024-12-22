import numpy as np
import os
from pathlib import Path
from src.trainers import BasicTrainer

from src.modules import (
    LinearLayer,
    LeakyReLULayer, 
    SigmoidLayer, 
    SoftmaxLayer,
    MSELoss, 
    CELoss,
)

"""
Dimension Names:
    N: number of datapoints
    D: number of dimensions in X data
    C: number of classes outputted by network
"""

class BaseNetwork:
    def __init__(self, neurons_per_layer, activations, loss, trainer):
        """
        Use neurons_per_layer to create layers of network
        """
        self.layers = []
        for i in range(len(neurons_per_layer)-1):
            self.layers.append(LinearLayer(neurons_per_layer[i], neurons_per_layer[i+1]))
            if i < len(activations): self.layers.append(activations[i])
        self.num_classes = neurons_per_layer[-1]

        self.loss = loss
        self.losses_per_epoch = []
        self.trainer = trainer(self)


    def forward(self, X, full_output=False):
        """
        Args:
            X: N x D numpy array
            with_cache: cache the prediction
        Return:
            prediction: N x C numpy array
        """
        curr = X.copy()
        for layer in self.layers:
            curr = layer.forward(curr)
        
        if not full_output: curr = np.argmax(curr, -1)
        return curr
    

    def train(self, X, y, batch_size=32, epochs=1000, learning_rate=1e-4, rel_tol=1e-5, loss_step=100):
        """
        Use trainer to train the model weights
        """
        self.trainer.train(X, y, batch_size, epochs, learning_rate, rel_tol, loss_step)


    def backward(self, y, prediction):
        """
        Args:
            y: N x C numpy array
            prediction: N x C numpy array
        """
        upstream = self.loss.backward(y, prediction)
        for layer in self.layers[::-1]:
            upstream = layer.backward(upstream)
    

    def gradient_update(self, learning_rate):
        """
        Args:
            learning_rate: scalar
        """
        for layer in self.layers:
            if layer.has_parameters: 
                layer.gradient_update(learning_rate)
    

    def get_loss(self, y, prediction):
        """
        Args:
            y: N x C numpy array
            prediction: N x C numpy array
        Return:
            loss: scalar
        """
        return self.loss.forward(y, prediction)
    

    def check_accuracy(self, y, inference):
        """
        Args:
            y: N numpy vector
            inference: N numpy vector
        """
        truths = np.count_nonzero(y == inference)
        return truths / len(y)
    
    
    def print_losses(self):
        for [epoch, loss] in self.losses_per_epoch:
            print("At epoch ", epoch, ", loss = ", loss)


    def print_params(self):
        for i in range(len(self.layers)):
            if self.layers[i].has_parameters:
                print("Layer", i, "has weights =", self.layers[i].weights)
                print("Layer", i, "has biases =", self.layers[i].biases)

    
    def print_gradients(self):
        i = len(self.layers)
        for layer in self.layers[::-1]:
            i -= 1
            if layer.has_parameters: 
                print("Layer", i, "gradients =", layer.gradients, "\nmax =", np.max(layer.gradients["weights"]))


    def save_params(self):
        cur_dir = f'saved_data/parameters'
        for i in range(len(self.layers)):
            if self.layers[i].has_parameters:
                np.save(f'{cur_dir}/weights_{i}.npy', self.layers[i].weights)
                np.save(f'{cur_dir}/biases_{i}.npy', self.layers[i].biases)

    
    def save_gradients(self):
        cur_dir = f'saved_data/gradients'
        for i in range(len(self.layers)):
            if self.layers[i].has_parameters:
                np.save(f'{cur_dir}/weights_{i}.npy', self.layers[i].gradients["weights"])
                np.save(f'{cur_dir}/biases_{i}.npy', self.layers[i].gradients["biases"])
                np.save(f'{cur_dir}/upstream_{i}.npy', self.layers[i].cached_gradients["upstream"])
                np.save(f'{cur_dir}/intermediate_{i}.npy', self.layers[i].cached_gradients["intermediate"])
                np.save(f'{cur_dir}/local_{i}.npy', self.layers[i].cached_gradients["local"])
                np.save(f'{cur_dir}/downstream_{i}.npy', self.layers[i].cached_gradients["downstream"])

    
    def load_params(self):
        cur_dir = f'saved_data/parameters'
        for i in range(len(self.layers)):
            if self.layers[i].has_parameters:
                self.layers[i].weights = np.load(f'{cur_dir}/weights_{i}.npy')
                self.layers[i].biases = np.load(f'{cur_dir}/biases_{i}.npy')



# MSE Loss + Sigmoid
class MSENetwork(BaseNetwork): 
    def __init__(self, neurons_per_layer, trainer=BasicTrainer):
        activations = [LeakyReLULayer(), LeakyReLULayer(), SigmoidLayer()]
        loss = MSELoss()
        super().__init__(neurons_per_layer, activations, loss, trainer)



# CE Loss + Softmax
class CENetwork(BaseNetwork): 
    def __init__(self, neurons_per_layer, trainer=BasicTrainer):
        activations = [LeakyReLULayer(), LeakyReLULayer(), SoftmaxLayer()]
        loss = CELoss()
        super().__init__(neurons_per_layer, activations, loss, trainer)
