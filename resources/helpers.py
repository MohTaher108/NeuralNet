# Helper functions such as shuffling data, splitting data, and multi-class classification

import numpy as np

def shuffle_data(features, labels):
    perm = np.random.permutation(features.shape[0])
    return features[perm], labels[perm]


def batch_split(features, labels, batch_size):
    num_datapoints = features.shape[0]
    num_arrays = num_datapoints // batch_size
    num_excess = num_datapoints % batch_size
    last_index = num_datapoints - num_excess

    if batch_size >= num_datapoints:
        return np.array([features]), np.array([labels])

    features_batches = np.array(np.vsplit(features[:last_index,:], num_arrays))
    labels_batches = np.array(np.split(labels[:last_index], num_arrays))

    # Accomodate for excess datapoints
    if num_excess > 0:
        excessFeatures = features[last_index:,:]
        excessLabels = labels[last_index:]
        
        # Choose random datapoints to pad array to be of length batch_size
        randomIndices = np.random.randint(num_datapoints, size=(batch_size - num_excess))
        excessFeatures = np.concatenate((excessFeatures, features[randomIndices]))
        excessLabels = np.concatenate((excessLabels, labels[randomIndices]))

        features_batches = np.concatenate((features_batches, excessFeatures[np.newaxis,:]), axis=0)
        labels_batches = np.concatenate((labels_batches, excessLabels[np.newaxis,:]), axis=0)

    return features_batches, labels_batches

def multi_class_classification(labels, num_labels_possible):
    aux_array = np.zeros((labels.shape[0], num_labels_possible))
    z = np.arange(labels.shape[0])
    aux_array[z, labels] = 1 # Index by 0,1,2,3 and labels to set correct values in array to 1
    return aux_array