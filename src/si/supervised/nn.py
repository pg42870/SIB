import numpy as np

from .model import Model
from scipy import signal
from abc import ABC, abstractmethod

from ..util.metrics import mse


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

    def setWeights(self, weights, bias):
        if (weights.shape != self.weights.shape):
            raise ValueError(f"Shape mismatch {weights.shape} and {self.weights.shape}")
        if (bias.shape != self.bias.shape):
            raise ValueError(f"Shapes mismatch {bias.shape} and {self.bias}")
        self.weights = weights
        self.bias = bias

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        """Computes dE/dw, dE/dB for a given output:error = dE/dY
        Returns input_error=dE/dX to feed the previous layer"""
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis= 0)
        input_error = np.dot(output_error, self.weights.T)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error

    def setweigths(self, weights, bias):
        self.weights = weights
        self.bias = bias


class Activation(Layer):

    def __init__(self, activation):
        self.activation = activation

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.input

    def backward(self, output_error, learning_rate):
        #learning_rate is not used because there is no 'lerarnable' parameters
        # only passed the error do the previous layer
        return np.multiply(self.activation.prime(self.input), output_error)



class NN(Model):

    def __init__(self, epochs = 1000, lr=0.001, verbose=True):
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

        self.layers = []
        self.loss = mse
       # self.loss_prime = mse_prime

    def add(self,layer):
        self.layers.append(layer)

    def fit(self, dataset):
        X, y = dataset.getXy()
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            output = X
            #forward propagation
            for layer in self.layers:
                output = layer.forward(output)

            #backwward propagation
            error = self.loss_prime(y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            #calculate average error on all samples
            err = self.loss(y, output)
            self.history[epoch] = err
            if self.verbose:
                print(f"epoch {epoch+1}/{self.epochs} error={err}")
        if not self.verbose:
            print(f"error={err}")
        self.is_fitted = True


    def predict(self, input_data):
        assert self.is_fitted, 'Model must be fit before predict'
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def cost(self, X=None, y=None):
        assert self.is_fitted, 'Model must be fit before predict'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        output = self.predict(X)
        return self.loss(y, output)


class Pooling2D():

    def __init__(self):
        return


    def forward(self, input):
        self.X_shape = input.shape
        n, h, w, d = input.shape
        h_out = (h - self.size) / self.stride + 1
        w_out = (w - self.size) / self.stride + 1

        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension')

        h_out, w_out = int(h_out), int(w_out)

        X_reshape = input.reshape(n * d, h, w, 1)

        self.X_col = im2col(X_reshape, self.size, self.size, padding=0, stride=self.stride)

        out, self.max_idxs = self.pool(self.X_col)

        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(3, 2, 0, 1)
        return out

    def backward(self, output_error, learning_raise):
        n, w, h, d = self.X_shape

        dX_col = np.zeros_like(self.X_col)
        dout_col = output_error.transpose(1, 2, 3, 0).ravel()

        dX = self.dpool(dX_col, dout_col, self.max_idx)
        dX = col2im(dX, (n * d, h, w, 1), self.size, self.size, padding=0, stride=self.stride)
        dX = dX.reshape(self.X_shape)

        return dX

