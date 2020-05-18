
import numpy as np
from cnn_utils import Cnn, Function
from cnn_operations import maxpoolBackward2d, convolutionBackward2d_Weights, convolutionBackward2d_FeatureMap, maxpool2d, convolution2d, convolutionBackward2d

cnn = Cnn()
f = Function()


class Model:
    def __init__(self, dropout_chance=0.0):
        self.dropout_chance = dropout_chance

        self.kernels, self.weights = cnn.CreateParameters(
            kernels=[(3, 3) for _ in range(36)],
            weight_matrices=[(56, 200),
                            (3, 56)]
        )

    """ forward pass """
    def forward(self, x, train=True):
        feature_maps = []
        max_feature_maps = []
        dense_layers = []

        if not train:
            self.dropout_chance = 0.0

        """ convolutional-layer 1 feature-map shape: (26, 26) """
        fm0 = f.relu(cnn.Conv2d(x, self.kernels[0].T.T))
        fm1 = f.relu(cnn.Conv2d(x, self.kernels[1].T.T))
        fm2 = f.relu(cnn.Conv2d(x, self.kernels[2].T.T))
        fm3 = f.relu(cnn.Conv2d(x, self.kernels[3].T.T))

        """ maxpool-layer 1 feature-map shape: (13, 13) """
        fm0_ = cnn.MaxPool2d(fm0, pooling_size=2)
        fm1_ = cnn.MaxPool2d(fm1, pooling_size=2)
        fm2_ = cnn.MaxPool2d(fm2, pooling_size=2)
        fm3_ = cnn.MaxPool2d(fm3, pooling_size=2)

        """ convolutional-layer 2, feature-map shape: (11, 11)  """
        fm4 = f.relu(cnn.Conv2d(fm0_, self.kernels[4].T.T))
        fm5 = f.relu(cnn.Conv2d(fm1_, self.kernels[5].T.T))
        fm6 = f.relu(cnn.Conv2d(fm2_, self.kernels[6].T.T))
        fm7 = f.relu(cnn.Conv2d(fm3_, self.kernels[7].T.T))

        fm8 = f.relu(cnn.Conv2d(fm0_, self.kernels[8].T.T))
        fm9 = f.relu(cnn.Conv2d(fm1_, self.kernels[9].T.T))
        fm10 = f.relu(cnn.Conv2d(fm2_, self.kernels[10].T.T))
        fm11 = f.relu(cnn.Conv2d(fm3_, self.kernels[11].T.T))

        """ maxpool-layer 2 feature-map shape: (5, 5) """
        fm4_ = cnn.MaxPool2d(fm4, pooling_size=2)
        fm5_ = cnn.MaxPool2d(fm5, pooling_size=2)
        fm6_ = cnn.MaxPool2d(fm6, pooling_size=2)
        fm7_ = cnn.MaxPool2d(fm7, pooling_size=2)

        fm8_ = cnn.MaxPool2d(fm8, pooling_size=2)
        fm9_ = cnn.MaxPool2d(fm9, pooling_size=2)
        fm10_ = cnn.MaxPool2d(fm10, pooling_size=2)
        fm11_ = cnn.MaxPool2d(fm11, pooling_size=2)

        feature_maps.extend([fm0, fm1, fm2, fm3, fm4, fm5, fm6, fm7, fm8, fm9, fm10, fm11]) 
        max_feature_maps.extend([fm0_, fm1_, fm2_, fm3_, fm4_, fm5_, fm6_, fm7_, fm8_, fm9_, fm10_, fm11_])

        """ dense-layers input: 544, output 3 """
        dense_in = cnn.flatten(max_feature_maps[4:])
        dense_hidden = f.relu(cnn.Linear(self.weights[0], dense_in))
        dense_hidden = cnn.DropOut(dense_hidden, chance=self.dropout_chance)
        dense_out = f.softmax(cnn.Linear(self.weights[1], dense_hidden))

        dense_layers.extend([dense_in, dense_hidden, dense_out])
        
        return dense_out, (x, self.kernels, self.weights, feature_maps, max_feature_maps, dense_layers)

    def print_shape(self, a):
        print(np.array(a).shape)

    """ backpropagation """
    def backward(self, prediction, target, parameters, lr=0.1):
        x, kernels, weights, feature_maps, max_feature_maps, dense_layers = parameters

        feature_maps_maxpool_2 = max_feature_maps[4:12]
        feature_maps_conv_2 = feature_maps[4:12]
        feature_maps_maxpool_1 = max_feature_maps[0:4]
        feature_maps_conv_1 = feature_maps[0:4]

        kernels_conv_2 = kernels[4:12]
        kernels_conv_1 = kernels[0:4]

        weight_out_delta = f.MSE(prediction, target, deriv=(True, f.softmax(dense_layers[2], deriv=True)))

        weight_hidden_loss = np.dot(weight_out_delta.T, weights[1])
        weight_hidden_delta = weight_hidden_loss.T * f.relu(dense_layers[1], deriv=True)

        ###########################################################################################################

        # input feature-map of convolution
        conv_in = feature_maps_maxpool_1

        # loss based on prior delta
        conv_2_loss = np.dot(weight_hidden_delta.T, weights[0]).reshape(8, 5, 5)                                        

        # backward maxpool (get 1 for every 'winning' index of maxpool operation, 0 for every 'losing')
        winning_maxpool_values_2 = []
        for i in range(len(feature_maps_conv_2)):
            winning_maxpool_values_2.append(maxpoolBackward2d(feature_maps_conv_2[i], 2))

        # rescale loss to size of pre-maxpool
        conv_2_delta = []
        for i in range(len(conv_2_loss)):
            zero_pad = np.zeros(winning_maxpool_values_2[i].shape)
            current_conv_2_delta = conv_2_loss[i].repeat(2, axis=0).repeat(2, axis=1)
            zero_pad[:current_conv_2_delta.shape[0], :current_conv_2_delta.shape[0]] = current_conv_2_delta
            current_conv_2_delta = zero_pad

            current_conv_2_delta = (current_conv_2_delta * winning_maxpool_values_2[i]) * f.relu(feature_maps_conv_2[i], deriv=True)

            conv_2_delta.append(current_conv_2_delta)

        part1_conv_2_delta = conv_2_delta[:int(len(conv_2_delta)/2)]
        part2_conv_2_delta = conv_2_delta[int(len(conv_2_delta)/2):]

        final_conv_2_delta = []
        for i in range(len(conv_in)):
            final_conv_2_delta.append(convolution2d(conv_in[i], part1_conv_2_delta[i].T.T).T.T)

        for i in range(len(conv_in)):
            final_conv_2_delta.append(convolution2d(conv_in[i], part2_conv_2_delta[i].T.T).T.T)

        ###########################################################################################################

        # input feature-map of convolution
        conv_in = x

        # loss based on prior delta
        part1_conv_2_delta = conv_2_delta[:int(len(conv_2_delta)/2)]
        part2_conv_2_delta = conv_2_delta[int(len(conv_2_delta)/2):]

        part1_kernels_conv_2 = kernels_conv_2[:int(len(kernels_conv_2)/2)]
        part2_kernels_conv_2 = kernels_conv_2[int(len(kernels_conv_2)/2):]

        conv_1_loss = []
        for i in range(len(part1_conv_2_delta)):
            part_1_loss = convolution2d(np.pad(part1_kernels_conv_2[i], 10, mode="constant"), part1_conv_2_delta[i].T.T)
            part_2_loss = convolution2d(np.pad(part2_kernels_conv_2[i], 10, mode="constant"), part2_conv_2_delta[i].T.T)

            conv_1_loss.append(part_1_loss + part_2_loss)

        # backward maxpool (creats mask: get 1 for every 'winning' index of maxpool operation, 0 for every 'losing')
        winning_maxpool_values_1 = []
        for i in range(len(feature_maps_conv_1)):
            winning_maxpool_values_1.append(maxpoolBackward2d(feature_maps_conv_1[i], 2))

        conv_1_delta = []
        for i in range(len(feature_maps_conv_1)):
            current_conv_1_delta = conv_1_loss[i]
            current_conv_1_delta = conv_1_loss[i].repeat(2, axis=0).repeat(2, axis=1)

            current_conv_1_delta = (current_conv_1_delta * winning_maxpool_values_1[i]) * f.relu(feature_maps_conv_1[i], deriv=True)

            conv_1_delta.append(current_conv_1_delta)

        final_conv_1_delta = []
        for i in range(len(conv_1_delta)):
            final_conv_1_delta.append(convolution2d(conv_in, conv_1_delta[i].T.T).T.T)

        #############################################################################################################

        self.weights[0] -= lr * np.dot(dense_layers[0], weight_hidden_delta.T).T
        self.weights[1] -= lr * np.dot(dense_layers[1], weight_out_delta.T).T

        for i in range(4):
            self.kernels[0:4][i] += lr * final_conv_1_delta[i]

        for i in range(8):
            self.kernels[4:12][i] += lr * final_conv_2_delta[i]
    
    def _gradient_clipping(self, matrix, theshold=2):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] >= theshold:
                    matrix[i][j] = 1
                if matrix[i][j] <= -theshold:
                    matrix[i][j] = -1
        
        return matrix