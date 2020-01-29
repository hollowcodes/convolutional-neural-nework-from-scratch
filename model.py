
import numpy as np
from cnn_utils import Cnn, Function

cnn = Cnn()
f = Function()


class Model:
    def __init__(self, dropout_chance=0.0):
        self.dropout_chance = dropout_chance

        self.kernels, self.weights = cnn.CreateParameters(
            kernels=[(3, 3) for _ in range(36)],
            weight_matrices=[(85, 144),
                            (35, 85),
                            (3, 35)]
        )

    """ forward pass """
    def forward(self, x, train=True):
        feature_maps = []
        dense_layers = []

        if not train:
            self.dropout_chance = 0.0

        """ 
            convolutional-layer 1 
            feature-map shape: (26, 26)
        """
        fm0 = f.relu(cnn.Conv2d(x, self.kernels[0]))
        fm1 = f.relu(cnn.Conv2d(x, self.kernels[1]))
        fm2 = f.relu(cnn.Conv2d(x, self.kernels[2]))
        fm3 = f.relu(cnn.Conv2d(x, self.kernels[3]))

        """
             maxpool-layer 1 
            feature-map shape: (13, 13)
        """
        fm0 = cnn.MaxPool2d(fm0, pooling_size=2)
        fm1 = cnn.MaxPool2d(fm1, pooling_size=2)
        fm2 = cnn.MaxPool2d(fm2, pooling_size=2)
        fm3 = cnn.MaxPool2d(fm3, pooling_size=2)

        """ 
            convolutional-layer 2
            feature-map shape: (11, 11) 
        """
        fm4 = f.relu(cnn.Conv2d(fm0, self.kernels[4]))
        fm5 = f.relu(cnn.Conv2d(fm0, self.kernels[5]))
        fm6 = f.relu(cnn.Conv2d(fm0, self.kernels[6]))
        fm7 = f.relu(cnn.Conv2d(fm0, self.kernels[7]))

        fm8 = f.relu(cnn.Conv2d(fm1, self.kernels[8]))
        fm9 = f.relu(cnn.Conv2d(fm1, self.kernels[9]))
        fm10 = f.relu(cnn.Conv2d(fm1, self.kernels[10]))
        fm11 = f.relu(cnn.Conv2d(fm1, self.kernels[11]))

        fm12 = f.relu(cnn.Conv2d(fm2, self.kernels[12]))
        fm13 = f.relu(cnn.Conv2d(fm2, self.kernels[13]))
        fm14 = f.relu(cnn.Conv2d(fm2, self.kernels[14]))
        fm15 = f.relu(cnn.Conv2d(fm2, self.kernels[15]))

        fm16 = f.relu(cnn.Conv2d(fm3, self.kernels[16]))
        fm17 = f.relu(cnn.Conv2d(fm3, self.kernels[17]))
        fm18 = f.relu(cnn.Conv2d(fm3, self.kernels[18]))
        fm19 = f.relu(cnn.Conv2d(fm3, self.kernels[19]))

        """ 
            maxpool-layer 2
            feature-map shape: (5, 5)
        """
        fm4 = cnn.MaxPool2d(fm4, pooling_size=2)
        fm5 = cnn.MaxPool2d(fm5, pooling_size=2)
        fm6 = cnn.MaxPool2d(fm6, pooling_size=2)
        fm7 = cnn.MaxPool2d(fm7, pooling_size=2)

        fm8 = cnn.MaxPool2d(fm8, pooling_size=2)
        fm9 = cnn.MaxPool2d(fm9, pooling_size=2)
        fm10 = cnn.MaxPool2d(fm10, pooling_size=2)
        fm11 = cnn.MaxPool2d(fm11, pooling_size=2)

        fm12 = cnn.MaxPool2d(fm12, pooling_size=2)
        fm13 = cnn.MaxPool2d(fm13, pooling_size=2)
        fm14 = cnn.MaxPool2d(fm14, pooling_size=2)
        fm15 = cnn.MaxPool2d(fm15, pooling_size=2)

        fm16 = cnn.MaxPool2d(fm16, pooling_size=2)
        fm17 = cnn.MaxPool2d(fm17, pooling_size=2)
        fm18 = cnn.MaxPool2d(fm18, pooling_size=2)
        fm19 = cnn.MaxPool2d(fm19, pooling_size=2)

        """ 
            convolutional-layer 3
            feature-map shape: (3, 3)
        """
        fm20 = f.relu(cnn.Conv2d(fm4, self.kernels[20]))
        fm21 = f.relu(cnn.Conv2d(fm5, self.kernels[21]))
        fm22 = f.relu(cnn.Conv2d(fm6, self.kernels[22]))
        fm23 = f.relu(cnn.Conv2d(fm7, self.kernels[23]))

        fm24 = f.relu(cnn.Conv2d(fm8, self.kernels[24]))
        fm25 = f.relu(cnn.Conv2d(fm9, self.kernels[25]))
        fm26 = f.relu(cnn.Conv2d(fm10, self.kernels[26]))
        fm27 = f.relu(cnn.Conv2d(fm11, self.kernels[27]))

        fm28 = f.relu(cnn.Conv2d(fm12, self.kernels[28]))
        fm29 = f.relu(cnn.Conv2d(fm13, self.kernels[29]))
        fm30 = f.relu(cnn.Conv2d(fm14, self.kernels[30]))
        fm31 = f.relu(cnn.Conv2d(fm15, self.kernels[31]))

        fm32 = f.relu(cnn.Conv2d(fm16, self.kernels[32]))
        fm33 = f.relu(cnn.Conv2d(fm17, self.kernels[33]))
        fm34 = f.relu(cnn.Conv2d(fm18, self.kernels[34]))
        fm35 = f.relu(cnn.Conv2d(fm19, self.kernels[35]))

        feature_maps.extend([fm0, fm1, fm2, fm3, fm4, fm5, fm6, fm7, fm8, fm9, fm10, fm11, fm12, fm13, fm14, fm15, fm16, fm17, fm18, fm19, 
                                  fm20, fm21, fm22, fm23, fm24, fm25, fm26, fm27, fm28, fm29, fm30, fm31, fm32, fm33, fm34, fm35])

        """
            dense-layers
            input: 544, output 3
        """
        dense_in = cnn.flatten(feature_maps[20:])
        dense_h0 = f.relu(cnn.Linear(self.weights[0], dense_in))
        dense_h0 = cnn.DropOut(dense_h0, chance=self.dropout_chance)
        dense_h1 = f.relu(cnn.Linear(self.weights[1], dense_h0))
        dense_h1 = cnn.DropOut(dense_h1, chance=self.dropout_chance)
        dense_out = f.sigmoid(cnn.Linear(self.weights[2], dense_h1))

        dense_layers.extend([dense_in, dense_h0, dense_h1, dense_out])
        
        return dense_out, (self.kernels, self.weights, feature_maps, dense_layers)

    """ backpropagation """
    def backward(self, prediction, target, parameters, lr=0.1):
        kernels, weights, feature_maps, dense_layers = parameters

        dense_out_delta = f.MSE(target, prediction, deriv=(True, f.sigmoid(dense_layers[3], deriv=True)))
        
        dense_h1_loss = np.dot(dense_out_delta.T, weights[2])
        dense_h1_delta = dense_h1_loss.T * f.relu(dense_layers[2], deriv=True)

        dense_h0_loss = np.dot(dense_h1_loss, weights[1])
        dense_h0_delta = dense_h0_loss.T * f.relu(dense_layers[1], deriv=True)

        self.weights[0] -= lr * np.dot(dense_layers[0], dense_h0_delta.T).T
        self.weights[1] -= lr * np.dot(dense_layers[1], dense_h1_delta.T).T
        self.weights[2] -= lr * np.dot(dense_layers[2], dense_out_delta.T).T

        # 16 x (3, 3) fms

        # 16 x (11, 11) fms

        # 4 x (26, 26) fms
        

