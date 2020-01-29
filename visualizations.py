
import matplotlib.pyplot as plt
import numpy as np

""" loads preprocessed dataset """
def load_dataset(dataset_path):
    return np.load(dataset_path, allow_pickle=True)

""" splits shufled dataset into its different classes """
def split_classes(dataset):
    cat_doodles, bus_doodles, umbrella_doodles = [], [], []
    for sample in dataset:
        if sample[1] == [[1], [0], [0]]:
            cat_doodles.append(sample)
        elif sample[1] == [[0], [1], [0]]:
            bus_doodles.append(sample)
        elif sample[1] == [[0], [0], [1]]:
            umbrella_doodles.append(sample)

    return cat_doodles, bus_doodles, umbrella_doodles

""" shows image matrices of certain class """
def show(batch, class_, amount=1):
    for sample in batch[:amount]:
        plt.matshow(sample[0])
        plt.title(class_)
        plt.show()


if __name__ == "__main__":
    """ loading the dataset from .npy file and concatenating the train, test, validation batches """
    dataset = load_dataset("dataset/dataset.npy")
    dataset = np.concatenate((dataset[0], dataset[1], dataset[0]))
    """ split the three classes from each other """
    cat_doodles, bus_doodles, umbrella_doodles = split_classes(dataset)

    """ show samples """
    show(cat_doodles, "cat", amount=2)
    show(bus_doodles, "bus", amount=2)
    show(umbrella_doodles, "umbrella", amount=2)