
import numpy as np
import random
import sys


""" loads raw dataset from numpy files """
def load_raw_dataset(cat_path, bus_path, umbrella_path, amount=2000):
    return np.load(cat_path)[:amount], np.load(bus_path)[:amount], np.load(umbrella_path)[:amount]

""" normalizes the matrix to a 0-1 range """
def normalize(batch):
    new_batch = []
    for sample in batch:
        new_sample = []
        for pixel in sample:
            new_sample.append(pixel / 255)
        new_batch.append(new_sample)

    return new_batch

""" reshapes the image lists to image matrices """
def reshape(batch):
    new_batch = []
    for sample in batch:
        new_batch.append(np.array(sample).reshape(28, 28))

    return new_batch

""" applies labels to batch """
def apply_label(batch, class_):
    label = []
    if class_ == "cat":
        label = [[1], [0], [0]]
    elif class_ == "bus":
        label = [[0], [1], [0]]
    elif class_ == "umbrella":
        label = [[0], [0], [1]]
    else:
        sys.exit("invalid class: '" , label, "'.")

    labeled_batch = []
    for sample in batch:
        full_sample = [sample, label]
        labeled_batch.append(full_sample)

    return labeled_batch

""" splits dataset into train, test and validation batch """
def split(dataset, testing_size=0.1, validation_size=0.1):
    test_size = int(np.round(len(dataset)*testing_size))
    val_size = int(np.round(len(dataset)*validation_size))
    train_set, test_set, validation_set = dataset[(test_size+val_size):], dataset[:test_size], dataset[test_size:(test_size+val_size)]

    return train_set, test_set, validation_set

""" save dataset to numpy file """
def save_dataset(dataset_path, dataset):
    np.save(dataset_path, dataset)

""" load preprocessed dataset """
def load_dataset(dataset_path):
    return np.load(dataset_path, allow_pickle=True) 
