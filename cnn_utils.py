
import numpy as np
from cnn_operations import convolution2d, maxpool2d


class Cnn:
	def __init__(self):
		pass

	def Conv2d(self, matrix, kernel):
		"""
			convolutional layer:
			 arg 1: input matrix
			 arg 2: weight kernel/filter

		"""
		
		return convolution2d(np.array(matrix), np.array(kernel))

	def MaxPool2d(self, matrix, pooling_size=3):
		""" 
			maxpool layer:
			 arg 1: input matrix
			 arg 2: pooling size

		"""

		return maxpool2d(matrix, pooling_size)

	def Linear(self, weights, vector):
		""" 
			dense layer
			arg 1: input vector (in matrix format: [[], [], ..., []])
			arg 2: weight matrix
		"""
		return np.dot(weights, vector)

	def DropOut(self, vector, chance=0.5):
		""" 
			dropout layer
			arg 1: input vector (in matrix format)
			arg 2: dropout chance
		"""
		percent = chance * 100
		for i in range(len(vector)):
			for j in range(len(vector[0])):
				rand = np.random.randint(101)
				if rand <= percent:
					vector[i][j] *= 0
				else:
					pass

		return vector

	def CreateParameters(self, kernels=[()], weight_matrices=[()]):
		""" 
			kernel and weight creation
			arg 1: list of all kernel shapes (indicated amount of kernels)
			arg 2: list of all weight matrix shapes (indicated amount of weight matrices)
		"""

		kernels_ = []
		for shape in kernels:
			kernels_.append(self._create_kernel(shape))

		weight_matrices_ = []
		for shape in weight_matrices:
			weight_matrices_.append(self._create_weight_matrix(shape))

		return kernels_, weight_matrices_

	def flatten(self, feature_maps):
		""" 
			feature-maps to dense layer convertion 
			arg: last feature-maps
		"""

		flat_layer = []
		for feature_map in feature_maps:

			for i in range(len(feature_map)):
				for j in range(len(feature_map[0])):
					flat_layer.append([feature_map[i][j]])
				
		return flat_layer

	def _create_kernel(self, kernel_size):
		""" creates kernel with given shape """

		return 2 * np.random.random(kernel_size) - 1

	def _create_weight_matrix(self, dimensions):
		""" creates weight matrix with given shape """

		return 2 * np.random.random(dimensions) - 1


class Function:
	def __init__(self):
		pass
	
	def sigmoid(self, x, deriv=False):
		""" sigmoid activation function """

		s = 1 / (1 + np.exp(-x))
		if deriv:
			return s * (1 - s)
		return s

	def relu(self, x, deriv=False):
		""" relu activation function """

		if deriv:
			return 1/2 * (1 + np.sign(x))
		return x/2 + 1/2 * x * np.sign(x)
		
	def softmax(self, x, deriv=False):
		""" softmax activation function """

		s = np.exp(x) / np.sum(np.exp(x))
		if deriv:
			return s * (1 - s)
		return s

	def MSE(self, y_true, y_prediction,  deriv=False, x=None, activation_function=None):
		""" mean-squared-error loss function """
		
		if deriv:
			return 2 * np.mean(np.subtract(y_true - y_prediction)) * activation_function(x, deriv=True)
		return np.mean(np.square(np.subtract(y_true, y_prediction)))


class Optim:
	def delta_SGD(self):
		""" stochastic gradient descent delta learning function """

		pass

