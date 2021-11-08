import numpy as np
from si.util.scale import StandardScaler as sc


class PCA:
    def __init__(self ,n_comp = 2, using="svd" ):
        self.n_comp = n_comp
        self.standardize = using

    def fit(self):
        pass


    def transform(self, dataset):
        scaled_feature = sc().fit_transform(dataset).X.T #normalizar com o StandardScaler
        if self.standardize == "svd":
            self.values, self.vectors, vh = np.linalg.svd(scaled_feature)
        else:
            covariance_matrix = np.cov(scaled_feature.T)  # matriz da covariancia
            self.values, self.vectors = np.linalg.eig(covariance_matrix)
            projection_matrix = (self.vectors.T[:][:self.n_comp]).T
            return scaled_feature.T.dot(projection_matrix)

        self.idxs = np.argsort(self.vectors)[::-1]  #idxs das colunas ordenadas por importância de compontes
        self.eigen_val, self.eigen_vect = self.vectors[self.idxs], self.values[:,self.idxs]  # colunas dos valores e dos vetores são reordenadas pelos idxs das colunas
        self.sub_set_vect = self.eigen_vect[:, :self.n_comp]  # gera um conjunto a partir dos vetores e values ordenados
        return scaled_feature.T.dot(self.sub_set_vect)


    def variance_explained(self):

        variance_explained = []
        for i in self.eigen_val:
            variance_explained.append((i / sum(self.eigen_val)) * 100)
        cumulative_variance_explained = np.cumsum(variance_explained)
        return cumulative_variance_explained

    def fit_transform(self, dataset):
        return self.transform(dataset)

def standardize_data(arr):
    '''
    This function standardize an array, its substracts mean value,
    and then divide the standard deviation.

    param 1: array
    return: standardized array
    '''
    rows, columns = arr.shape
    X = arr

    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)

    for column in range(columns):

        mean = np.mean(X[:, column])
        std = np.std(X[:, column])
        tempArray = np.empty(0)

        for element in X[:, column]:
            tempArray = np.append(tempArray, ((element - mean) / std))

        standardizedArray[:, column] = tempArray

    return standardizedArray