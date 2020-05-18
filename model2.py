
import numpy as np
from cnn_utils import Cnn, Function
from cnn_operations import maxpoolBackward2d, convolutionBackward2d_Weights, convolutionBackward2d_FeatureMap, maxpool2d, convolution2d, convolutionBackward2d

cnn = Cnn()
f = Function()


class Model:
    def __init__(self, dropout_chance=0.0):
        self.dropout_chance = dropout_chance

        self.kernels, self.weights = cnn.CreateParameters(
            kernels=[(3, 3) for _ in range(6)],
            weight_matrices=[(60, 100),
                            (3, 60)]
        )

    """ forward pass """
    def forward(self, x, train=True):
        feature_maps = []
        max_feature_maps = []
        dense_layers = []

        if not train:
            self.dropout_chance = 0.0

        """ convolutional-layer 1 feature-map shape: (26, 26) """
        fm0 = f.relu(cnn.Conv2d(x, self.kernels[0]))
        fm1 = f.relu(cnn.Conv2d(x, self.kernels[1]))

        """ maxpool-layer 1 feature-map shape: (13, 13) """
        fm0_ = cnn.MaxPool2d(fm0, pooling_size=2)
        fm1_ = cnn.MaxPool2d(fm1, pooling_size=2)

        """ convolutional-layer 2, feature-map shape: (11, 11)  """
        fm2 = f.relu(cnn.Conv2d(fm0_, self.kernels[2]))
        fm3 = f.relu(cnn.Conv2d(fm0_, self.kernels[3]))
        fm4 = f.relu(cnn.Conv2d(fm1_, self.kernels[4]))
        fm5 = f.relu(cnn.Conv2d(fm1_, self.kernels[5]))

        """ maxpool-layer 2 feature-map shape: (5, 5) """
        fm2_ = cnn.MaxPool2d(fm2, pooling_size=2)
        fm3_ = cnn.MaxPool2d(fm3, pooling_size=2)
        fm4_ = cnn.MaxPool2d(fm4, pooling_size=2)
        fm5_ = cnn.MaxPool2d(fm5, pooling_size=2)

        feature_maps.extend([fm0, fm1, fm2, fm3, fm4, fm5]) 
        max_feature_maps.extend([fm0_, fm1_, fm2_, fm3_, fm4_, fm5_])

        """ dense-layers input: 544, output 3 """
        dense_in = cnn.flatten(max_feature_maps[2:])
        dense_hidden = f.sigmoid(cnn.Linear(self.weights[0], dense_in))
        dense_hidden = cnn.DropOut(dense_hidden, chance=self.dropout_chance)
        dense_out = f.sigmoid(cnn.Linear(self.weights[1], dense_hidden))

        dense_layers.extend([dense_in, dense_hidden, dense_out])
        
        return dense_out, (x, self.kernels, self.weights, feature_maps, max_feature_maps, dense_layers)

    def print_shape(self, a):
        print(np.array(a).shape)

    """ backpropagation """
    def backward(self, prediction, target, parameters, lr=0.1):
        x, kernels, weights, feature_maps, max_feature_maps, dense_layers = parameters

        feature_maps_maxpool_2 = max_feature_maps[2:]
        feature_maps_conv_2 = feature_maps[2:]
        feature_maps_maxpool_1 = max_feature_maps[:2]
        feature_maps_conv_1 = feature_maps[:2]

        kernels_conv_2 = kernels[2:]
        kernels_conv_1 = kernels[:2]

        weight_out_delta = f.MSE(prediction, target, deriv=(True, f.sigmoid(dense_layers[2], deriv=True)))

        weight_hidden_loss = np.dot(weight_out_delta.T, weights[1])
        weight_hidden_delta = weight_hidden_loss.T * f.sigmoid(dense_layers[1], deriv=True)

        ###########################################################################################################

        """ GRADIENTS FOR CONV-2 """

        conv_in = feature_maps_maxpool_1
        conv_2_loss = np.dot(weight_hidden_delta.T, weights[0]).reshape(4, 5, 5)                                        

        """ gradients for conv2-kernel 0 """
        max_mask_2_0 = maxpoolBackward2d(feature_maps_conv_2[0], 2)
        conv_2_loss_0 = conv_2_loss[0]
        zero_pad = np.zeros(max_mask_2_0.shape)
        conv_2_delta_0 = conv_2_loss_0.repeat(2, axis=0).repeat(2, axis=1)
        zero_pad[:conv_2_delta_0.shape[0], :conv_2_delta_0.shape[0]] = conv_2_delta_0
        conv_2_delta_0 = zero_pad
        conv_2_delta_0 = (conv_2_delta_0 * max_mask_2_0) * f.relu(feature_maps_conv_2[0], deriv=True)

        """ gradients for conv2-kernel 1 """
        max_mask_2_1 = maxpoolBackward2d(feature_maps_conv_2[1], 2)
        conv_2_loss_1 = conv_2_loss[1]
        zero_pad = np.zeros(max_mask_2_1.shape)
        conv_2_delta_1 = conv_2_loss_1.repeat(2, axis=0).repeat(2, axis=1)
        zero_pad[:conv_2_delta_1.shape[0], :conv_2_delta_1.shape[1]] = conv_2_delta_1
        conv_2_delta_1 = zero_pad
        conv_2_delta_1 = (conv_2_delta_1 * max_mask_2_1) * f.relu(feature_maps_conv_2[1], deriv=True)

        """ gradients for conv2-kernel 2 """
        max_mask_2_2 = maxpoolBackward2d(feature_maps_conv_2[2], 2)
        conv_2_loss_2 = conv_2_loss[2]
        zero_pad = np.zeros(max_mask_2_2.shape)
        conv_2_delta_2 = conv_2_loss_2.repeat(2, axis=0).repeat(2, axis=1)
        zero_pad[:conv_2_delta_2.shape[0], :conv_2_delta_2.shape[0]] = conv_2_delta_2
        conv_2_delta_2 = zero_pad
        conv_2_delta_2 = (conv_2_delta_2 * max_mask_2_2) * f.relu(feature_maps_conv_2[2], deriv=True)

        """ gradients for conv2-kernel 3 """
        max_mask_2_3 = maxpoolBackward2d(feature_maps_conv_2[3], 2)
        conv_2_loss_3 = conv_2_loss[3]
        zero_pad = np.zeros(max_mask_2_3.shape)
        conv_2_delta_3 = conv_2_loss_3.repeat(2, axis=0).repeat(2, axis=1)
        zero_pad[:conv_2_delta_3.shape[0], :conv_2_delta_3.shape[0]] = conv_2_delta_3
        conv_2_delta_3 = zero_pad
        conv_2_delta_3 = (conv_2_delta_3 * max_mask_2_3) * f.relu(feature_maps_conv_2[3], deriv=True)

        final_conv_2_delta_0 = convolution2d(conv_in[0], conv_2_delta_0)
        final_conv_2_delta_1 = convolution2d(conv_in[0], conv_2_delta_1)
        final_conv_2_delta_2 = convolution2d(conv_in[1], conv_2_delta_2)
        final_conv_2_delta_3 = convolution2d(conv_in[1], conv_2_delta_3)

        ###########################################################################################################

        """ GRADIENTS FOR CONV-1 """

        conv_in = x

        conv_1_loss_0_part_1 = convolution2d(np.pad(kernels_conv_2[0], 10, mode="constant"), conv_2_delta_0)
        conv_1_loss_0_part_2 = convolution2d(np.pad(kernels_conv_2[1], 10, mode="constant"), conv_2_delta_1)
        conv_1_loss_0 = conv_1_loss_0_part_1 + conv_1_loss_0_part_2

        conv_1_loss_1_part_1 = convolution2d(np.pad(kernels_conv_2[2], 10, mode="constant"), conv_2_delta_2)
        conv_1_loss_1_part_2 = convolution2d(np.pad(kernels_conv_2[3], 10, mode="constant"), conv_2_delta_3)
        conv_1_loss_1 = conv_1_loss_1_part_1 + conv_1_loss_1_part_2

        """ gradients for conv1-kernel 0 """
        max_mask_1_0 = maxpoolBackward2d(feature_maps_conv_1[0], 2)
        zero_pad = np.zeros(max_mask_1_0.shape)
        conv_1_delta_0 = conv_1_loss_0.repeat(2, axis=0).repeat(2, axis=1)
        zero_pad[:conv_1_delta_0.shape[0], :conv_1_delta_0.shape[0]] = conv_1_delta_0
        conv_1_delta_0 = zero_pad
        conv_1_delta_0 = (conv_1_delta_0 * max_mask_1_0) * f.relu(feature_maps_conv_1[0], deriv=True)

        """ gradients for conv1-kernel 1 """
        max_mask_1_1 = maxpoolBackward2d(feature_maps_conv_1[1], 2)
        zero_pad = np.zeros(max_mask_1_1.shape)
        conv_1_delta_1 = conv_1_loss_1.repeat(2, axis=0).repeat(2, axis=1)
        zero_pad[:conv_1_delta_1.shape[0], :conv_1_delta_1.shape[0]] = conv_1_delta_1
        conv_1_delta_1 = zero_pad
        conv_1_delta_1 = (conv_1_delta_1 * max_mask_1_1) * f.relu(feature_maps_conv_1[1], deriv=True)

        final_conv_1_delta_0 = convolution2d(conv_in, conv_1_delta_0)
        final_conv_1_delta_1 = convolution2d(conv_in, conv_1_delta_1)

        #############################################################################################################

        self.weights[0] += lr * np.dot(dense_layers[0], weight_hidden_delta.T).T
        self.weights[1] += lr * np.dot(dense_layers[1], weight_out_delta.T).T

        self.kernels[0] += lr * final_conv_1_delta_0
        self.kernels[1] += lr * final_conv_1_delta_1

        self.kernels[2] += lr * final_conv_2_delta_0
        self.kernels[3] += lr * final_conv_2_delta_1
        self.kernels[4] += lr * final_conv_2_delta_2
        self.kernels[5] += lr * final_conv_2_delta_3
    
    def _gradient_clipping(self, matrix, theshold=2):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] >= theshold:
                    matrix[i][j] = 1
                if matrix[i][j] <= -theshold:
                    matrix[i][j] = -1
        
        return matrix