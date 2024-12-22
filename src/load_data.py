from keras.datasets import mnist
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os


def load_data_penguins(category_name):
    le = LabelEncoder()
    penguins = load_penguins()
    X = np.array(penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']])
    y = le.fit_transform(np.array(penguins[category_name]))

    X, y = remove_nans(X, y)
    X /= np.max(X, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test, le


def pull_MNIST(num_points=0):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    if num_points != 0: X_train, X_test, y_train, y_test = reduce_dataset_size(X_train, X_test, y_train, y_test, num_points)
    return X_train, X_test, y_train, y_test


def remove_nans(X, y):
    indices = np.unique(np.where(~np.isnan(X))[0])
    return X[indices], y[indices]


def reduce_dataset_size(X_train, X_test, y_train, y_test, new_num_points, train_proportion=0.9):
    indices_train = np.random.choice(X_train.shape[0], size=int(train_proportion * new_num_points))
    indices_test = np.random.choice(X_test.shape[0], size=int((1 - train_proportion) * new_num_points))
    return X_train[indices_train], X_test[indices_test], y_train[indices_train], y_test[indices_test]


def save_point(X_train, y_train, index):
    X_point = X_train[index][np.newaxis]
    y_point = y_train[index][np.newaxis]
    
    cur_dir = f'saved_data/point'
    np.save(f'{cur_dir}/X_point.npy', X_point)
    np.save(f'{cur_dir}/y_point.npy', y_point)


def load_point():
    cur_dir = f'saved_data/point'
    X_point = np.load(f'{cur_dir}/X_point.npy')
    y_point = np.load(f'{cur_dir}/y_point.npy')
    return X_point, y_point
