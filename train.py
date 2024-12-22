from src import (
    load_data_penguins,
    MSENetwork,
    CENetwork,
)
import time

INPUT_NEURONS = 4
OUTPUT_NEURONS = 3
HIDDEN_NEURONS = [6,]

def evaluate_model(my_net, X_test, y_test):
    inference = my_net.forward(X_test)
    print("Accuracy: ", my_net.check_accuracy(y_test, inference))


def train_model(Network=MSENetwork, batch_size=16, epochs=200, learning_rate=2e-1, rel_tol=1e-5, loss_step=50):
    X_train, X_test, y_train, y_test, _ = load_data_penguins("species")
    neurons_per_layer = [INPUT_NEURONS, *HIDDEN_NEURONS, OUTPUT_NEURONS]
    my_net = Network(neurons_per_layer)

    start = time.time()
    my_net.train(X_train, y_train, batch_size, epochs, learning_rate, rel_tol, loss_step)
    my_net.print_losses()
    print("Time elapsed during training: ", time.time() - start)

    evaluate_model(my_net, X_test, y_test)
    my_net.save_gradients()
    my_net.save_params()
    
    
train_model()
