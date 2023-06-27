from myNet import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1)) # using -1 basically leaves the remaining length in its place (so length(array) / n_samples)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Create and train image
num_layers = 3
num_neurons_per_layer = [12, 12, 10]
num_dimensions = 64
neural_net = NeuralNet(num_layers, num_neurons_per_layer, num_dimensions)
neural_net.train(X_train, y_train, epochs=1001)

# Plot loss vs epochs
x = np.linspace(0, (len(neural_net.losses)-1)*50, len(neural_net.losses))
plt.plot(x, neural_net.losses)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

# Print classification report from predictions
predicted = neural_net.feedForward(X_test)
print("predicted.shape = ", predicted.shape)
print(
    f"Classification report:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)