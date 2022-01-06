from numpy.core.numeric import identity
from numpy.lib.function_base import gradient
from .model import Model
from ..util.metrics import mse
import numpy as np
from ..util.util import sigmoid

class LogisticRegression(Model):

    def __init__(self, gd=False, epochs=1000, lr=0.001):
        '''Linear regression Model
        epochs: number of epochs
        lr: learning rate for GD
        '''
        super(LogisticRegression, self).__init__()
        self.theta = None
        self.epochs = epochs
        self.lr = lr

    def fit(self, dataset):
        X, Y = dataset.getXy()
        X = np.hstack(
            (np.ones((X.shape[0], 1)), X))  # acrescentar o nosso x só com 1 que corresponde ao termo independente
        self.X = X
        self.Y = Y
        # Closed form or GD
        self.train(X, Y)  # implement closed train form (see notes)
        self.is_fitted = True

    def train(self, X, Y):
        n = X.shape[1]
        self.hsitory = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            self.theta -= self.lr * gradient
            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self, X):
        p = self.probability(X)
        res = 1 if p >= 0.5 else 0
        return res

    def cost(self):
        pass


class LogisticRegressionReg:

	' Regressão Logística com regularização'

	def __init__(self, epochs=1000, lr=0.1, lbd=1):
		super(LogisticRegressionReg, self).__init__()
		self.theta = None
		self.epochs = epochs
		self.lr = lr
		self.lbd = lbd  # lbd = lambda

	def fit(self, dataset):
		X, y = dataset.getXy()
		X = add_intersect(X)
		########
		# Só é necessário para fazer o score (cost) caso não queiram dar os dados
		self.X = X
		self.Y = y
		##########
		# closed form or GD
		self.train(X, y)
		self.is_fitted = True

	def train(self, X, y):
		m, n = X.shape
		self.history = {}
		self.theta = np.zeros(n)
		for epoch in range(self.epochs):
			z = np.dot(X, self.theta)
			h = sigmoid(z)
			grad = np.dot(X.T, (h - y)) / y.size
			reg = (self.lbd / m) * self.theta[1:]  ###### parentesis
			grad[1:] = grad[1:] + reg
			self.theta -= self.lr * grad
			self.history[epoch] = [self.theta[:], self.cost()]

	def predict(self, x):
		assert self.is_fitted, 'Model must be fit before predicting'
		hs = np.hstack(([1], x))
		p = sigmoid(np.dot(self.theta, hs))
		if p >= 0.5: res = 1
		else: res = 0
		return res

	def cost(self, X=None, y=None, theta=None):
		X = add_intersect(X) if X is not None else self.X
		y = y if y is not None else self.Y
		theta = theta if theta is not None else self.theta
		m = X.shape[0]
		h = sigmoid(np.dot(X, theta))
		cost = (-y * np.log(h) - (1 - y) * np.log(1 - h))
		reg = np.dot(theta[1:], theta[1:]) * self.lbd / (2 * m)
		res = np.sum(cost) / m
		return res + reg