import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derive_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def MSE_loss(truth, prediction):
    return np.mean((truth - prediction) ** 2)


class NeuralNet:
    def __init__(self):
        # 6 weights and 3 biases for a neural network with 2 hidden layer of 2;1 neurons with input size of 2 features
        self.weights = np.random.normal(size=6)
        self.biases = np.random.normal(size=3)
    
    # Feed forward through entire neural network
    def feedForward(self, data, return_all=False):
        h1 = sigmoid(self.weights[0] * data[0] + self.weights[1] * data[1] + self.biases[0])
        h2 = sigmoid(self.weights[2] * data[0] + self.weights[3] * data[1] + self.biases[1])
        o1 = sigmoid(self.weights[4] * h1 + self.weights[5] * h2 + self.biases[2])
        if return_all:
            return h1, h2, o1
        else:
            return o1   

    # Train over 1000 epochs using SGD with a learning rate of 0.1
    def train(self, data, learning_rate=0.1, epochs=1000):
        features = data[:,:-1]
        labels = data[:,-1:].flatten()

        for epoch in range(epochs):
            for datapoint, label in zip(features, labels):
                h1, h2, o1 = self.feedForward(datapoint, return_all=True)

                dL_do1 = -2 * (label - o1)

                # Neuron o1
                derive_sigmoid_o1 = o1 * (1 - o1)
                do1_dw5 = h1 * derive_sigmoid_o1
                do1_dw6 = h2 * derive_sigmoid_o1
                do1_db3 = derive_sigmoid_o1

                do1_dh1 = self.weights[4] * derive_sigmoid_o1
                do1_dh2 = self.weights[5] * derive_sigmoid_o1

                # Neuron h1
                derive_sigmoid_h1 = h1 * (1 - h1)
                dh1_dw1 = datapoint[0] * derive_sigmoid_h1
                dh1_dw2 = datapoint[1] * derive_sigmoid_h1
                dh1_db1 = derive_sigmoid_h1

                # Neuron h2
                derive_sigmoid_h2 = h2 * (1 - h2)
                dh2_dw3 = datapoint[0] * derive_sigmoid_h2
                dh2_dw4 = datapoint[1] * derive_sigmoid_h2
                dh2_db2 = derive_sigmoid_h2
                
                # Neuron o1
                dL_w5 = dL_do1 * do1_dw5
                dL_w6 = dL_do1 * do1_dw6
                dL_b3 = dL_do1 * do1_db3

                self.weights[4] -= learning_rate * dL_w5
                self.weights[5] -= learning_rate * dL_w6
                self.biases[2] -= learning_rate * dL_b3

                # Neuron h1
                dL_w1 = dL_do1 * do1_dh1 * dh1_dw1
                dL_w2 = dL_do1 * do1_dh1 * dh1_dw2
                dL_b1 = dL_do1 * do1_dh1 * dh1_db1

                self.weights[0] -= learning_rate * dL_w1
                self.weights[1] -= learning_rate * dL_w2
                self.biases[0] -= learning_rate * dL_b1

                # Neuron h2
                dL_w3 = dL_do1 * do1_dh2 * dh2_dw3
                dL_w4 = dL_do1 * do1_dh2 * dh2_dw4
                dL_b2 = dL_do1 * do1_dh2 * dh2_db2

                self.weights[2] -= learning_rate * dL_w3
                self.weights[3] -= learning_rate * dL_w4
                self.biases[1] -= learning_rate * dL_b2

            if epoch % 50 == 0:
                prediction = np.apply_along_axis(self.feedForward, 1, data)
                loss = MSE_loss(labels, prediction)
                print("Epoch %d loss: %.3f" % (epoch, loss))


Alice = np.array([133, 65, 1], dtype='float')
Bob = np.array([160, 72, 0], dtype='float')
Charlie = np.array([152, 70, 0], dtype='float')
Diana = np.array([120, 60, 1], dtype='float')
data = np.array([Alice, Bob, Charlie, Diana])

data_averaged = np.mean(data[:,:-1], axis=0)
data[:,:-1] -= data_averaged
print("data = ", data)

neural_net = NeuralNet()
neural_net.train(data)