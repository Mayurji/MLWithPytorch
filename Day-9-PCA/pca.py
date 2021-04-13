import torch
from sklearn.datasets import load_iris
import seaborn as sb
import matplotlib.pyplot as plt

class pca:
    def __init__(self, n_components):
        """
        :param n_components: Number of principal components the data should be reduced too.
        """
        self.components = n_components

    def fit_transform(self, X):
        """
        * Centering our inputs with mean
        * Finding covariance matrix using centered tensor
        * Finding eigen value and eigen vector using torch.eig()
        * Sorting eigen values in descending order and finding index of high eigen values
        * Using sorted index, get the eigen vectors
        * Tranforming the Input vectors with n columns into PCA components with reduced dimension
        :param X: Input tensor with n columns.
        :return: Output tensor with reduced principal components
        """
        centering_X = X - torch.mean(X, dim=0)
        covariance_matrix = torch.mm(centering_X.T, centering_X)/(centering_X.shape[0] - 1)
        eigen_values, eigen_vectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigen_sorted_index = torch.argsort(eigen_values[:,0],descending=True)
        eigen_vectors_sorted = eigen_vectors[:,eigen_sorted_index]
        component_vector = eigen_vectors_sorted[:,0:self.components]
        transformed = torch.mm(component_vector.T, centering_X.T).T
        return transformed

if __name__ == '__main__':
    data = load_iris()
    X = torch.tensor(data.data,dtype=torch.double)
    y = torch.tensor(data.target)
    pca = pca(n_components=2)
    pca_vector = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))
    sb.scatterplot(pca_vector[:, 0], pca_vector[:, 1], hue=y, s=60, palette='icefire')
    plt.show()
