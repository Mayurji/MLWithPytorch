import torch
from utility import euclidean_distance
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
"""
K-Medoids also known as Partitioned Around Medoids.
"""
class PAM:
    def __init__(self, k=2):
        """
        :param k: Number of clusters to be formed using Medoids
        """
        self.k = k

    def random_medoids(self, X):
        """
        Similar to KMeans, selecting a random samples from dataset as medoids
        :param X: Input tensor
        :return: For iris dataset, three medoids are selected.
        """
        n_samples, n_features = X.shape[0], X.shape[1]
        medoids = torch.zeros((self.k, n_features))
        for i in range(self.k):
            idx = torch.randperm(len(X))[1]
            medoid = X[idx]
            medoids[i] = medoid

        return  medoids

    def closest_medoid(self, sample, medoids):
        """
        Calculate distance between each sample and every medoids
        :param sample: Data point
        :param medoids: Similar to centroid in KMeans.
        :return: Assigining medoid to each sample
        """
        closest_i = None
        closest_distance = float('inf')
        for i, medoid in enumerate(medoids):
            distance = euclidean_distance(sample, medoid)
            if distance < closest_distance:
                closest_i = i
                closest_distance = distance
        return closest_i

    def create_clusters(self, X, medoids):
        """
        Creating clusters after assigning samples to each medoid
        :return:
        """
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            medoid_i = self.closest_medoid(sample, medoids)
            clusters[medoid_i].append(sample_i)

        return clusters

    def calculate_cost(self, X, clusters, medoids):
        """
        Total distance between samples and their medoid
        :param clusters: Three medoids with samples assigned to each of them
        :return: Total distance as mentioned above
        """
        cost = 0
        for i, cluster in enumerate(clusters):
            medoid = medoids[i]
            for sample_i in cluster:
                cost += euclidean_distance(X[sample_i], medoid)

        return cost

    def get_non_medoids(self, X, medoids):
        """
        Mediods are points in cluster acts reference for all other points(non-medoids)
        to find distance between them.
        :return: all the data point which are not medoids.
        """
        non_medoids = []
        for sample in X:
            if not sample in medoids:
                non_medoids.append(sample)

        return non_medoids

    def get_cluster_label(self, clusters, X):
        """
        Assigning each sample as index to a medoid.
        """
        y_pred = torch.zeros(X.shape[0])
        for cluster_i in range(len(clusters)):
            cluster = clusters[cluster_i]
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i

        return y_pred

    def predict(self, X):
        """
        Do Partitioning Around Medoids and return the cluster labels
        * First, randomly selection medoids
        * Create cluster based on medoids selected and samples
        * Cost(distance) of the existing cluster and the samples in it.
        * Iterate, until we find the least cost with best medoids.
        * Find all non-medoids
        :return: Predicting medoid for test sample or a data point.
        """

        medoids = self.random_medoids(X)
        clusters = self.create_clusters(X, medoids)
        cost = self.calculate_cost(X, clusters, medoids)

        while True:
            best_medoids = medoids
            lowest_cost = cost
            for medoid in medoids:
                non_medoids = self.get_non_medoids(X, medoids)
                # Calculate the cost when swapping medoid and samples
                for sample in non_medoids:
                    # Swap sample with the medoid
                    new_medoids = medoids.clone()
                    new_medoids[medoids == medoid][:4] = sample
                    # Assign samples to new medoids
                    new_clusters = self.create_clusters(X, new_medoids)
                    # Calculate the cost with the new set of medoids
                    new_cost = self.calculate_cost(X, new_clusters, new_medoids)
                    # If the swap gives us a lower cost we save the medoids and cost
                    if new_cost < lowest_cost:
                        lowest_cost = new_cost
                        best_medoids = new_medoids
            # If there was a swap that resultet in a lower cost we save the
            # resulting medoids from the best swap and the new cost
            if lowest_cost < cost:
                cost = lowest_cost
                medoids = best_medoids
            else:
                break

        final_clusters = self.create_clusters(X, medoids)
        # Return the samples cluster indices as labels
        return self.get_cluster_label(final_clusters, X)


if __name__ == '__main__':
    data = load_iris()
    X = data.data
    y = data.target
    # Cluster the data using K-Medoids
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y)
    clf = PAM(k=3)
    y_pred = clf.predict(X)
    print(accuracy_score(y_pred, y))


