import numpy as np
from si.util.scale import StandardScaler as sc


class PCA:
    def __init__(self, n_comp=2, using="svd"):
        self.n_comp = n_comp
        self.using = using

    def fit(self):
        pass

    def transform(self, dataset):
        scaled_feature = sc().fit_transform(dataset).X.T  # normalize with StandardScaler

        if self.using == "svd":
            self.vectors, self.values, vh = np.linalg.svd(scaled_feature)
        else:
            covariance_matrix = np.cov(scaled_feature.T)  # matriz da covariancia
            self.vectors, self.values = np.linalg.eig(covariance_matrix)

        # idxs of the sorted columns by feature importance
        self.idxs = np.argsort(self.vectors)[::-1]

        # columns of the values and vectors are reorganized by the columns idxs
        self.eigen_vec, self.eigen_val = self.vectors[self.idxs], self.values[:,
                                                                  self.idxs]

        projection_matrix = (self.eigen_vect.T[:][:self.n_comp]).T
        return scaled_feature.T.dot(projection_matrix)

    def variance_explained(self):
        variance_explained = []
        for i in self.eigen_val:
            variance_explained.append((i / sum(self.eigen_val)) * 100)
        return variance_explained

    def fit_transform(self, dataset):
        return self.transform(dataset)
