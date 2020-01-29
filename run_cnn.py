
import numpy as np
from tqdm import tqdm
import time

from model import Model
from cnn_utils import Cnn, Function
from preprocessing_utils import load_dataset
from utils import *

f = Function()


class Run:
    def __init__(self, epochs=10, lr=1e-5, decay_rate=1e-2, dropout_chance=0.0):
        self.epochs = epochs
        self.lr = lr
        self.decay_rate = decay_rate
        self.dropout_chance = dropout_chance

        self.model = Model(dropout_chance=self.dropout_chance)

    """ train neural network """
    def train(self, train_set, validation_set):
        start = time.time()

        for epoch in tqdm(range(self.epochs), ncols=75, desc="progress"):
            epoch_loss = []

            for idx in range(len(train_set)):
                sample = train_set[idx][0]
                target = train_set[idx][1]

                prediction, parameters = self.model.forward(sample)
                
                loss = f.MSE(target, prediction)

                self.model.backward(prediction, target, parameters, lr=self.lr)

                epoch_loss.append(loss)

            val_acc = round(self.evaluate(validation_set), 4)
            progress(self.epochs, (epoch + 1), round(loss, 4), val_acc, (time.time() - start))

    """ evaluate neural network (for validation-/test-set) """
    def evaluate(self, dataset):
        total = len(dataset)
        correct = 0

        for idx in range(len(dataset)):
            sample = dataset[idx][0]
            target = dataset[idx][1]

            prediction, _ = self.model.forward(sample, train=False)

            same = True
            for i in range(0, 2):
                if np.round(prediction[0]) != target[0]:
                    same = False
                    break
                
            if same:
                correct += 1
        
        return correct / total

    """ test neural network """
    def test(self, test_set):
        accuracy = round(self.evaluate(test_set), 4)
        print("\naccuracy after",colored(self.epochs, "cyan", attrs=['bold']), "epochs:", colored(accuracy, "cyan", attrs=['bold']), "%")


if __name__ == "__main__":
    run = Run(epochs=20, 
              lr=1e-2, 
              decay_rate=1e-2, 
              dropout_chance=0.5)

    dataset = load_dataset("dataset/dataset.npy")
    train_set, test_set, validation_set = dataset[0], dataset[1], dataset[2]

    run.train(train_set, validation_set)
    run.test(test_set)