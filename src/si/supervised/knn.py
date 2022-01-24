import numpy as np
from ..util.util import l2_distance
from ..util.metrics import accuracy_score
from .model import Model


class KNN(Model):
    def __init__(self, num_neighbors, classification=True):
        """
        Abstract class defining an interface fro supervised learning models
        """
        super(KNN).__init__()
        self.k = num_neighbors
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self, x):
        """Returns the distance of the k neighbors that are closer"""
        distances = l2_distance(x, self.dataset.X)

        # sort the distances
        sorted_index = np.argsort(distances)
        return sorted_index[:self.k]

    def predict(self, x):
        assert self.is_fitted, "Model must be fit before predict"

        # gets the neighbors
        neighbors = self.get_neighbors(x)

        # transforms it in a list of y values that corresponded to the neighbors
        values = self.dataset.Y[neighbors].tolist()

        # from the list counts the occurrences and returns the values of the occurrences that have more counts
        prediction = max(set(values), key=values.count)
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.T)
        return accuracy_score(self.dataset.Y, y_pred)
