# Helper functions such as shuffling data and splitting data

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