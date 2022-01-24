import numpy as np
from ..util.util import l2_distance


class KMeans:

    def __init__(self, k: int, n_iter=800, distance=l2_distance) -> None:
        self.k = k
        self.max_iter = n_iter
        self.centroids = None
        self.distance = distance

    def fit(self, dataset):
        """randomly select k centroids"""
        x = dataset.X

        self._min = np.min(x, axis=0)
        self._max = np.max(x, axis=0)

    def init_centroids(self, dataset):
        """ Initiate the centroids
        gets the np array where the values will the randomly generated to be in a normal distribution
        """
        x = dataset.X

        self.centroids = np.array(
            [np.random.uniform(
                low=self._min[1], high=self._max[i], size=(self.k,)
            ) for i in range(x.shape[1])]).T

    def get_closest_centroid(self, x):
        """returns the centroid id of the closest centroid"""
        dist = self.distance(x, self.centroids)
        closest_centroids_index = np.argmin(dist, axis=0)
        return closest_centroids_index

    def transform(self, dataset):
        self.init_centroids(dataset)
        print(self.centroids)

        X = dataset.X
        changed = True
        count = 0
        old_idxs = np.zeros(X.shape[0])

        while changed and count < self.max_iter:
            idxs = np.apply_along_axis(self.get_closest_centroid, axis=0,
                                       arr=X.T)  # applies get_closest_centroid for all the x

            cent = []
            for i in range(self.k):
                cent.append(np.mean(X[idxs == i], axis=0))

            self.centroids = np.array(cent)
            count += 1
            changed = not np.all(old_idxs == idxs)  # tests if all arguments are True
            old_idxs = idxs
        return self.centroids, old_idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
