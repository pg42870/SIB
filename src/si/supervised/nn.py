import numpy as np

from .model import Model
from scipy import signal

class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error, learning_rate):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """Fully connect layer"""
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.zeros((1,output_size))

    def setWeights(self, weigths, bias):
        if (weights.shape != self.weights-shape):
            raise ValueError(f"Shape mismatch {weights.shape} and {self.weights.shape}")
        if (bias.shape != self.bias.shape):
            raise ValueError(f"Shapes mismatch {bias.shape} and {self.bias}")
        self.weights = weigths
        self.bias = bias

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        raise NotImplementedError


class Activation(Layer):

    def __init__(self, activation):
        self.activation = activation

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.input

    def backward(self, output_error, learning_rate):
        raise NotImplementedError


class NN(Model):

    def __init__(self, epochs = 1000, lr=0.001, verbose=True):
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

        self.layer = []
        self.loss = mse
        self.loss_prime = mse_prime

    def add(self,layer):
        self.layers.append(layer)

    def fit(self):
        raise NotImplementedError

    def predict(self, input_data):
        assert self.is_fitted, 'Model must be fit before predict'
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def cost(self, X=None, y=None):
        assert self.is_fitted, 'Model must be fit before precict'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        output = self.predict(X)
        return self.loss(y,output)

