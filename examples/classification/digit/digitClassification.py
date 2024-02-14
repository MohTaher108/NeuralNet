# Digit classification on 8x8 bw images from sklearn

from myNet import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split


# load and flatten the images
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1)) # using -1 basically leaves the remaining length in its place (so length(array) / n_samples)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Normalize values so softmax doesn't explode (divides by 16)
X_train /= np.max(X_train)
X_test /= np.max(X_train)

# Create neural net
input_size = np.size(X_train[0])
output_size = 10
num_neurons_per_layer = [input_size, 12, output_size] # Input shape, layers' shapes, output shape
neural_net = NeuralNet(num_neurons_per_layer)

# Train neural net
epochs, batch_size, learning_rate, loss_step = 3001, 32, 2e-3, 100 #2e-3
neural_net.train(X_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, loss_step=loss_step)

# Plot loss vs epochs
x = np.linspace(0, (len(neural_net.losses)-1)*loss_step, len(neural_net.losses))
plt.plot(x, neural_net.losses)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

# Print classification report from predictions
predicted = neural_net.feed_forward(X_test)
print(
    f"\n\n\n\n\nClassification report:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)