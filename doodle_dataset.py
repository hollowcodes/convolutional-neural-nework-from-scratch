
import numpy as np
from preprocessing_utils import *


class DoodleDataset:
    def __init__(self, raw_dataset_paths: tuple=(), preprocessed_dataset_path: str=""):
        self.raw_dataset_paths = raw_dataset_paths
        self.preprocessed_dataset_path = preprocessed_dataset_path

    def preprocess(self):
        """ dataset paths """
        cat_path, bus_path, umbrella_path = self.raw_dataset_paths

        """ loading the dataset batches (cat, bus, umbrella) from .npy files """
        print("[*] loading raw dataset")
        cat_dataset, bus_dataset, umbrella_dataset = load_raw_dataset(cat_path, bus_path, umbrella_path, amount=100)

        """ normalizing the samples to [0, 1]"""
        print("[*] normalizing samples")
        cat_dataset, bus_dataset, umbrella_dataset = normalize(cat_dataset), normalize(bus_dataset), normalize(umbrella_dataset)

        """ reshaping the samples from list to matrix """
        print("[*] reshaping to 2d matrix")
        cat_dataset, bus_dataset, umbrella_dataset = reshape(cat_dataset), reshape(bus_dataset), reshape(umbrella_dataset)

        """ applying labels to the samples of the three dataset batches """
        print("[*] applying labels")
        cat_dataset, bus_dataset, umbrella_dataset = apply_label(cat_dataset, "cat"), apply_label(bus_dataset, "bus"), apply_label(umbrella_dataset, "umbrella")

        """ concatenating the three batches and shuffling """
        print("[*] concatenating and shuffling")
        dataset = np.concatenate((np.array(cat_dataset), np.array(bus_dataset), np.array(umbrella_dataset)))
        np.random.shuffle(dataset)

        """ splitting the dataset into training, testing and validation batch """
        print("[*] splitting into train, test, validation")
        train_batch, test_batch, validation_batch = split(dataset, testing_size=0.1, validation_size=0.2)
        dataset = np.array([train_batch, test_batch, validation_batch])

        print("[*] finished dataset preprocessing")

        return dataset

    """ saving the dataset to .npy file """
    def save(self):
        print("[*] saving preprocessed dataset")
        save_dataset(self.preprocessed_dataset_path, dataset)


if __name__ == "__main__":
    doodleDataset = DoodleDataset(raw_dataset_paths=("dataset/doodles/full_numpy_bitmap_cat.npy", 
                                                     "dataset/doodles/full_numpy_bitmap_bus.npy", 
                                                     "dataset/doodles/full_numpy_bitmap_umbrella.npy"),
                                  preprocessed_dataset_path="dataset/dataset.npy")
                                   
    dataset = doodleDataset.preprocess()
    doodleDataset.save()