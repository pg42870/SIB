from abc import ABC, abstractmethod
from si.util.util import l2_distance
from si.util.metrics import accuracy_score
import numpy as np

class Model(ABC):
    def __init__(self, num_neighbors):
        """
        Abstract class definihg an interface fro supervised learning models
        """
        supper(KNN).__init__()
        self.k = num_neighbors

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self, x):
        distances = l2_distance(x, self.dataset.X)
        sorted_index = np.argsort(distances) #ordenar as distancias
        return sorted_index[:self.k] #devolver a distancia dos k vizinhos que estao mais proximos

    def predict(self, x):
        assert self.is_fitted, "Model must be fit before predict"
        neighbors = self.get_neighbors(x) #vai buscar quais os vizinhos
        values = self.dataset.Y[neighbors].tolist() #transforma em lista os valores do Y correspondentes aos vizinhos
        prediction = max(set(values), key=values.count) #vai ver a lista conta as ocorrencias e devolve os valores com as ocorrencias mais altas
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.T)
        return accuracy_score(self.dataset.Y, y_pred)

