from myNet import NeuralNet
import numpy as np
import matplotlib.pyplot as plt

Alice = np.array([133, 65, 1], dtype='float')
Bob = np.array([160, 72, 0], dtype='float')
Charlie = np.array([152, 70, 0], dtype='float')
Diana = np.array([120, 60, 1], dtype='float')
data = np.array([Alice, Bob, Charlie, Diana])

data_averaged = np.mean(data[:,:-1], axis=0)
data[:,:-1] -= data_averaged

num_layers = 4
num_neurons_per_layer = [2, 2, 2, 1]
num_dimensions = 2
neural_net = NeuralNet(num_layers, num_neurons_per_layer, num_dimensions)
features = data[:,:-1]
labels = data[:,-1:].flatten()
neural_net.train(features, labels, epochs=1001)

x = np.linspace(0, 1000, 21)
plt.plot(x, neural_net.losses)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()


Emily = np.array([128, 63], dtype='float')
Emily -= data_averaged
Frank = np.array([155, 68], dtype='float')
Frank -= data_averaged

Emily_prediction = neural_net.feedForward(Emily)
print("Emily %.3f " % Emily_prediction)
if np.rint(Emily_prediction) == 1:
    print("Emily must be female")
elif np.rint(Emily_prediction) == 0:
    print("Emily must be male")

Frank_prediction = neural_net.feedForward(Frank)
print("Frank %.3f " % Frank_prediction)
if np.rint(Frank_prediction) == 1:
    print("Emily must be female")
elif np.rint(Frank_prediction) == 0:
    print("Emily must be male")