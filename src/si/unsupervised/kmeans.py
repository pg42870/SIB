import numpy as np
from si.util.util import l2_distance
from copy import copy


class KMeans:

    def __init__(self, k: iter, n_iter=800):
        self.k = k
        self.max_iter = n_iter
        self.centroids = None
        self.distance = l2_distance

    def fit(self, dataset):
        """randomly select k centroids"""
        x = dataset.X
        self._min = np.min(x, axis=0)
        self._max = np.max(x, axis=0)

    def init_centroids(self, dataset):
        """
        iniciar os centroides
        geramos um np array onde os valores serao gerados de forma randomica de modo a terem uma distribuicao uniforme

        """
        x = dataset.X
        self.centroids = np.array(
            [np.random.uniform(
                low=self._min[1], high=self._max[i], size=(self.k,)
            ) for i in range(x.shape[1])]).T
        #rng = np.random.default_rng()
        #self.centroids = rng.choice(copy(dataset.X), size=self.k, replace=False, p=None, axis=0)

    def get_closest_centroid(self, x):
        "retorna o id do centroide mais proximo"
        dist = self.distance(x, self.centroids)
        closest_centroids_index = np.argmin(dist, axis=0)
        return closest_centroids_index

    def transform(self,dataset):
        """ """
        self.init_centroids(dataset)
        print(self.centroids)
        X = dataset.X
        changed = True
        count = 0
        old_idxs = np.zeros(X.shape[0])
        while changed and count < self.max_iter:
            idxs = np.apply_along_axis(self.get_closest_centroid, axis=0, arr=X.T) #aplica a funcao get_closest_centroid a todo o x
            cent = []
            for i in range(self.k):
                cent.append(np.mean(X[idxs == i], axis=0))
            self.centroids = np.array(cent)
            count += 1
            changed = np.all(old_idxs == idxs) #testa se todos os argumentos sao True
            old_idxs = idxs
        return self.centroids, old_idxs


    def fit_transform(self,dataset):
        self.fit(dataset)
        return self.transform(dataset)





