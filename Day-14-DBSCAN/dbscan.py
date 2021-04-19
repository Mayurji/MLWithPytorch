"""
Checkout Density Based Spectral Clustering Blag:
https://blog.dominodatalab.com/topology-and-density-based-clustering/

- Compared to centroid-based clustering like k-means, density-based clustering works by
identifying “dense” clusters of points, allowing it to learn clusters of arbitrary shape
and identify outliers in the data.
"""
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets

class DBScan:
    def __init__(self, eps = 2.5, min_points=30):
        """
        eps - radius distance around which a cluster is considered.
        min_points -  Number of points to be present inside the radius
        (check out density reachable or border points from blog to understand how cluster points are considered)
        """
        self.eps = eps
        self.minimum_points = min_points

    def euclidean_distance(self, x1, x2):
        """
        :param x1: input tensor
        :param x2: input tensor
        :return: distance between tensors
        """
        return torch.cdist(x1, x2)

    def direct_neighbours(self, sample):
        """
        :param sample: Sample whose neighbors needs to be identified
        :return: all the neighbors within eps distance
        """
        neighbors = []
        idxs = torch.arange(self.X.shape[0])
        for i, _sample in enumerate(self.X[idxs != sample]):

            distance = self.euclidean_distance(self.X[sample].unsqueeze(0), _sample.unsqueeze(0))
            if distance < self.eps:
                neighbors.append(i)

        return torch.tensor(neighbors)

    def density_neighbors(self, sample, neighbors):
        """
        Recursive method which expands the cluster until we have reached the border
        of the dense area (density determined by eps and min_samples)

        :param sample: Sample whose border points to be identified
        :param neighbors: samples and its neighbors within eps distance
        :return: It updates the number of points assigned to each cluster, by finding
        border points and its relative points. In a sense, it expands cluster.
        """
        cluster = [sample]
        for neighbor_i in neighbors:
            if not neighbor_i in self.visited_samples:
                self.visited_samples.append(neighbor_i)
                self.neighbors[neighbor_i] = self.direct_neighbours(neighbor_i)

                if len(self.neighbors[neighbor_i]) >= self.minimum_points:
                    expanded_cluster = self.density_neighbors(
                        neighbor_i, self.neighbors[neighbor_i])
                    cluster = cluster + expanded_cluster
                else:
                    cluster.append(neighbor_i)

        return cluster

    def get_cluster_label(self):
        """
        :return: assign cluster label based on expanded clusters
        """
        labels = torch.zeros(self.X.shape[0]).fill_(len(self.clusters))
        for cluster_i, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_i

        return labels

    def predict(self, X):
        """
        :param X: input tensor
        :return: predicting the labels os samples depending on its distance from clusters
        """
        self.X = X
        self.clusters = []
        self.visited_samples = []
        self.neighbors = {}
        n_samples = X.shape[0]

        for sample_i in range(n_samples):
            if sample_i in self.visited_samples:
                continue
            self.neighbors[sample_i] = self.direct_neighbours(sample_i)
            if len(self.neighbors[sample_i]) >= self.minimum_points:
                self.visited_samples.append(sample_i)
                new_cluster = self.density_neighbors(
                    sample_i, self.neighbors[sample_i])
                self.clusters.append(new_cluster)

        cluster_labels = self.get_cluster_label()
        return cluster_labels

if __name__ == '__main__':
    iris = load_iris()
    torch.manual_seed(0)
    X = torch.tensor(iris.data, dtype=torch.float)
    y = torch.tensor(iris.target)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    dbscan = DBScan(eps=0.25, min_points=20)
    ypred = dbscan.predict(x_train)
    print(f'Accuracy Score: {accuracy_score(y_train, ypred)}')
