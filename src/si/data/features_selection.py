import numpy as np
from copy import copy
import warnings
from scipy import stats
from scipy.stats import f

from .dataset import Dataset


class VarianceThreshold:
    def __init__(self, threshold=0):
        if threshold < 0:
            warnings.warn("The threshold must be a non-negative value.")
            threshold = 0
        self.threshold = threshold

    def fit(self, dataset):
        #recebe o dataset e guarda numa variavel as variancias dos X
        X = dataset.X
        self._var= np.var(X, axis=0)

    def transform(self, dataset, inline=False):
        X = dataset.X
        cond = self._var > self.threshold #array de booleanos
        idxs = [i for i in range(len(cond)) if cond[i]]
        X_trans = X[:,idxs] #sob as colunas features
        xnames = [dataset.xnames[i] for i in idxs]
        if inline:
            dataset.X = X_trans
            dataset.xnames = xnames
            return dataset
        else:
            return Dataset(copy(X_trans), copy(dataset.Y), xnames, copy(dataset.yname))

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)



def f_regression(dataset):
    # calcula o coeficiente de correlacao com dois graus de liberdade
    X = dataset.X
    Y = dataset.Y
    corr_coef = np.array([stats.pearsonr(X[:,i],Y)[0] for i in range(X.shape[1])])
    deg_freedom = Y.size - 2 #número de determinações independentes menos o número de parâmetros estatísticos a serem avaliados na população
    corr_coef_squared = corr_coef ** 2
    F = corr_coef_squared / (1 - corr_coef_squared) * deg_freedom
    p = f.sf(F, 1, deg_freedom)
    return F,p


def f_classification(dataset):
    # devolve quais as features que contribuem mais para que tenham a mesma media
    X = dataset.X
    y = dataset.Y

    args = [X[y == a, :] for a in np.unique(y)]
    F, p = stats.f_oneway(*args)
    return F, p



class SelectKBest:
    """
    Escolhe as melhores features
    """
    def __init__(self, k: int, score_funcs):
        if score_funcs in (f_classification, f_regression):
            self._func = score_funcs

        if k > 0:
            self.k = k
        else:
            warnings.warn("Invalid feature number, k must be greater than 0")

    def fit(self,dataset):
        self.F, self.p = self._func(dataset)

    def transform(self, dataset, inline=False):
        X = dataset.X
        xnames = dataset.xnames
        feat_selection = sorted(np.argsort(self.F)[-self.k:]) #ordenado por ordem crescente, nos queremos os maiores valores por isso vao estar no fim
        x = X[:, feat_selection]
        xnames = [xnames[feat] for feat in feat_selection]

        if inline:
            dataset.X = x
            dataset.xnames = xnames
            return dataset
        else:
            return Dataset(x, copy(dataset.Y), xnames, copy(dataset.yname))

    def fit_transform(self,dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)

