import numpy as np
from scipy import stats
import warnings


class VarianceThreshould:
    def __init__(self, threshold=0):
        if threshold < 0:
            warnings.warn("The threshold must be a non-negative value.")
        self.threshold = threshold

    def fit(selfself, dataset):
        X = dataset.X
        self._var= np.var(X, axis=0)

    def transform(self, dataset, inline=False):
        X = dataset.X
        cond = self._var > self.threshold #array de booleanos
        indxs = [i for i in range(len(cond)) if cond[i]]
        X_trans = X[:,idxs]
        xnames = [dataset._xnames[i] for i in idxs]
        if inline:
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset
        else:
            from .dataset import Dataset
            return Dataset(copy(X_trans),) #nao acabei esta parte

class SelectKBest:
    def __int__(self, dataset):
        self.F, self.p =


