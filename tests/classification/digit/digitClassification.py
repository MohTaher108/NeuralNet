# Digit classification on 8x8 bw images from sklearn

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

# Normalize values so softmax doesn't explode (divides by 16)
X_train /= np.max(X_train)
X_test /= np.max(X_train)

# Create and train image
input_size = np.size(X_train[0])
output_size = 10
num_neurons_per_layer = [input_size, 12, output_size] # Input shape, layers' shapes, output shape
neural_net = NeuralNet(num_neurons_per_layer)
epochs, learning_rate, loss_step = 3001, 0.15, 10
neural_net.train(X_train, y_train, GD_type="GD", epochs=epochs, learning_rate=learning_rate, loss_step=loss_step)

# predicted = neural_net.feed_forward(X_test)
# print(
#     f"Classification report:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )

# neural_net.reset_weights()
# neural_net.train(X_train, y_train, GD_type="SGD", epochs=epochs, learning_rate=learning_rate, loss_step=loss_step)
# neural_net.print_parameters()

# Plot loss vs epochs
# x = np.linspace(0, (len(neural_net.losses)-1)*loss_step, len(neural_net.losses))
# plt.plot(x, neural_net.losses)
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.show()

# Print classification report from predictions
print("\n\n\n\n\n")
predicted = neural_net.feed_forward(X_test)
print(
    f"Classification report:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)





# epochs_list = [101, 101, 101, 101, 101, 101, 101, 101, 101, 101]
# loss_step = 100
# accuracies = []

# for j in range(5):
#     accuracies.append([])
#     neural_net.init_weights()
#     for i in range(len(epochs_list)):
#         print("iteration ", i+1)
#         neural_net.train(X_train, y_train, GD_type="SGD", epochs=epochs_list[i], learning_rate=learning_rate, loss_step=loss_step)
#         accuracy = metrics.accuracy_score(y_test, neural_net.feed_forward(X_test)) * 1e2
#         # print("Accuracy = ", accuracy)
#         accuracies[j].append(accuracy)

# print("accuracies = ", accuracies)
# # print("average accuracy = ", np.mean(accuracies))