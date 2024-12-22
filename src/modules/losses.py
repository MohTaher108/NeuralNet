import numpy as np

"""
Dimension Names:
    N: number of datapoints
    C: number of classes outputted by network
"""


class MSELoss:    
    def forward(self, truth, prediction):
        """
        Args:
            truth: N x C numpy array
            prediction: N x C numpy array
        Return:
            loss: scalar
        """
        loss = np.mean(np.sum((truth - prediction) ** 2, axis=-1))
        return loss


    def backward(self, truth, prediction):
        """
        Args:
            truth: N x C numpy array
            prediction: N x C numpy array
        Return:
            upstream: N x C x 1 numpy array
        """
        upstream = -2 * (truth - prediction) / prediction.shape[0] # divide by N to account for np.mean() in MSE Loss
        upstream = upstream[:,:,np.newaxis]
        return upstream
    


class CELoss:    
    def forward(self, truth, prediction):
        """
        Args:
            truth: N x C numpy array
            prediction: N x C numpy array
        Return:
            loss: scalar
        """
        loss = -1 * np.mean(np.sum(truth * np.log(prediction + 1e-15), axis=-1))
        return loss
    

    def backward(self, truth, prediction):
        """
        Args:
            truth: N x C numpy array
            prediction: N x C numpy array
        Return:
            upstream: N x C x 1 numpy array
        """
        upstream = prediction - truth
        upstream = upstream[:,:,np.newaxis]
        return upstream
