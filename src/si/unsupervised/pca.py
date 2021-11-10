import numpy as np
from si.util.scale import StandardScaler as sc


class PCA:
    def __init__(self, n_comp=2, using="svd"):
        self.n_comp = n_comp
        self.using = using

    def fit(self):
        pass


    def transform(self, dataset):
        scaled_feature = sc().fit_transform(dataset).X.T #normalizar com o StandardScaler
        if self.using == "svd":
            self.vectors, self.values, vh = np.linalg.svd(scaled_feature)
        else:
            covariance_matrix = np.cov(scaled_feature.T)  # matriz da covariancia
            self.vectors, self.values = np.linalg.eig(covariance_matrix)

        self.idxs = np.argsort(self.vectors)[::-1]  #idxs das colunas ordenadas por importância de compontes
        self.eigen_vec, self.eigen_val = self.vectors[self.idxs], self.values[:, self.idxs]  # colunas dos valores e dos vetores são reordenadas pelos idxs das colunas
        projection_matrix = (self.eigen_vect.T[:][:self.n_comp]).T
        return scaled_feature.T.dot(projection_matrix)


    def variance_explained(self):
        variance_explained = []
        for i in self.eigen_val:
            variance_explained.append((i / sum(self.eigen_val)) * 100)
        return variance_explained

    def fit_transform(self, dataset):
        return self.transform(dataset)

