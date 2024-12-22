import numpy as np

class BasicTrainer:
    def __init__(self, network):
        self.network = network


    def train(self, X, y, batch_size, epochs, learning_rate, rel_tol, loss_step):
        if y.ndim == 1: y = make_multi_class(y, self.network.num_classes)
        for epoch in range(epochs):
            X_batches, y_batches = batch_split(X, y, batch_size)
            for X_batch, y_batch in zip(X_batches, y_batches):
                self.gradient_descent(X_batch, y_batch, learning_rate)
            if (epoch % loss_step == 0 or epoch == epochs-1) and self.check_loss(X, y, epoch, rel_tol):
                break


    def gradient_descent(self, X, y, learning_rate=0.1):
        prediction = self.network.forward(X, full_output=True)
        self.network.backward(y, prediction)
        self.network.gradient_update(learning_rate)


    def check_loss(self, X, y, epoch, rel_tol=1e-4):
        prediction = self.network.forward(X, full_output=True)
        loss = self.network.get_loss(y, prediction)
        self.network.losses_per_epoch.append((epoch, loss))

        if len(self.network.losses_per_epoch) > 1:
            prev_loss = self.network.losses_per_epoch[-2][1]
            if np.abs(loss - prev_loss) / prev_loss < rel_tol:
                return True
        
        return False
    
    

def make_multi_class(y, num_classes):
    """
    Go from each label being a single value between 0 and num_classes to an array of num_classes from 0 to 1
    """
    aux_array = np.zeros((y.shape[0], num_classes))
    z = np.arange(y.shape[0])
    aux_array[z, y] = 1 # Index by 0,1,2,3 and labels to set correct values in array to 1
    return aux_array


def batch_split(X, y, batch_size):
    """
    Split X and y into lists of size batch_size
    Accommodate for excess datapoints by padding last lists with random points from X and y
    """
    num_datapoints = X.shape[0]
    num_arrays = num_datapoints // batch_size
    num_excess = num_datapoints % batch_size
    last_index = num_datapoints - num_excess

    if batch_size >= num_datapoints:
        return np.array([X]), np.array([y])

    X_batches = np.array(np.vsplit(X[:last_index,:], num_arrays))
    y_batches = np.array(np.split(y[:last_index], num_arrays))

    # Accomodate for excess datapoints
    if num_excess > 0:
        excessX = X[last_index:,:]
        excessy = y[last_index:]
        
        # Choose random datapoints to pad array to be of length batch_size
        randomIndices = np.random.randint(num_datapoints, size=(batch_size - num_excess))
        excessX = np.concatenate((excessX, X[randomIndices]))
        excessy = np.concatenate((excessy, y[randomIndices]))

        X_batches = np.concatenate((X_batches, excessX[np.newaxis,:]), axis=0)
        y_batches = np.concatenate((y_batches, excessy[np.newaxis,:]), axis=0)

    return X_batches, y_batches
