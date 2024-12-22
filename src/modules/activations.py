import numpy as np

"""
Dimension Names:
    N: number of datapoints
    K: number of neurons in layer
"""


class LeakyReLULayer:
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha: scalar (ReLU negative scaling)
            has_parameters: boolean (indicates if layer has parameters to be updated)
        """
        self.alpha = alpha
        self.has_parameters = False
        self.input = None
    

    def forward(self, input):
        """
        Return:
            output: same as input shape
        """
        self.input = input # Cache for backprop
        output = input.copy()
        output[output <= 0] *= self.alpha
        return output


    def backward(self, upstream):
        """
        Return:
            new_upstream: same as upstream shape
        """
        new_upstream = upstream.copy()
        new_upstream[self.input <= 0] *= self.alpha

        upstream_addition = np.ones(upstream.shape)
        upstream_addition[self.input <= 0] = self.alpha

        return new_upstream



class SigmoidLayer:
    def __init__(self):
        """
        Args:
            has_parameters: boolean (indicates if layer has parameters to be updated)
        """
        self.has_parameters = False
    

    def forward(self, input):
        """
        Return:
            output: same as input shape
        """
        # Numerically stable sigmoid
        output = np.where(
            input >= 0,
            1 / (1 + np.exp(-input)), # positives
            np.exp(input) / (1 + np.exp(input)) # negatives
        )
        # output = 1 / (1 + np.exp(-input))
        return output


    def backward(self, upstream):
        """
        Return:
            new_upstream: same as upstream shape
        """
        new_upstream = upstream * (1 - upstream)
        return new_upstream
    


class SoftmaxLayer:
    def __init__(self):
        """
        Args:
            has_parameters: boolean (indicates if layer has parameters to be updated)
        """
        self.has_parameters = False
    

    def forward(self, input):
        """
        Return:
            output: same as input shape
        """
        input_stable = input - np.max(input, axis=-1, keepdims=True) # Prevent an exploding value
        exponent = np.exp(input_stable)
        output = exponent / np.sum(exponent, axis=-1, keepdims=True)
        return output


    def backward(self, upstream):
        """
        Return:
            new_upstream: same as upstream shape
        """
        new_upstream = upstream.copy()
        return new_upstream
