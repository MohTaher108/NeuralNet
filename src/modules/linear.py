import numpy as np

"""
Dimension Names:
    N: number of datapoints
    C: number of neurons in current layer
    B: number of neurons in behind layer
    A: number of neurons in ahead layer
"""

INDEXINDEX = 0


class LinearLayer:
    def __init__(self, input_size, num_neurons):
        """
        Args:
            weights: B x C numpy array
            biases: C numpy array
            has_parameters: boolean (indicates if layer has parameters to be updated)
            input: N x B numpy array (cached for backprop)
            gradients: map with gradients for weights and biases
        """
        self.weights = np.random.normal(size=(input_size, num_neurons))
        self.biases = np.random.normal(size=(num_neurons))
        self.has_parameters = True

        self.input = None
        self.gradients = {"biases": None, "weights": None}
        self.cached_gradients = {"upstream": None, "intermediate": None, "local": None, "downstream": None}
    
    
    def forward(self, input):
        """
        Args:
            input: N x B numpy array
        Return:
            output: N x C numpy array
        """
        self.input = input # Cache for backprop
        output = input @ self.weights + self.biases
        return output
    

    def backward(self, upstream):
        """
        Args:
            upstream: N x C x A numpy array
        Uses:
            input: N x B numpy array
            weights: B x C numpy array
        Caches:
            biases gradient: N x C numpy array
            weights gradient: N x B x C numpy array
        Return:
            downstream: N x B x C numpy array
        """
        self.gradients["biases"] = np.mean(upstream, axis=-1)

        temp = self.gradients["biases"][:,np.newaxis,:]
        self.gradients["weights"] = temp * self.input[:,:,np.newaxis]

        downstream = temp * self.weights[np.newaxis,:,:]

        self.cached_gradients["upstream"] = upstream
        self.cached_gradients["intermediate"] = self.weights
        self.cached_gradients["local"] = self.input
        self.cached_gradients["downstream"] = downstream

        return downstream


    def gradient_update(self, learning_rate):
        """
        Args:
            learning_rate: scalar
        """
        self.biases -= learning_rate * np.mean(self.gradients["biases"], axis=0)
        self.weights -= learning_rate * np.mean(self.gradients["weights"], axis=0)
