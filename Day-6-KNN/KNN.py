import torch
from scipy.stats import mode
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNN:
    def __init__(self, k, X):
        """
        :param k: Number of Neighbors
        """
        self.k = k

    def distance(self, point_1, point_2, default='euclidean', p=2):
        if default == 'euclidean':
            return torch.norm(point_1 - point_2, 2, 0)
        elif default == 'manhattan':
            return torch.sum(torch.abs(point_1 - point_2))
        elif default == "minkowski":
            return torch.pow(torch.sum(torch.abs(point_1 - point_2)**p), 1/p)
        else:
            raise ValueError("Unknown similarity distance type")

    def fit_predict(self, X, y, item):
        """
        * Iterate through each datapoints (item/y_test) that needs to be classified
        * Find distance between all train data points and each datapoint (item/y_test)
          using euclidean distance
        * Sort the distance using argsort, it gives indices of the y_test
        * Find the majority label whose distance closest to each datapoint of y_test.


        :param X: Input tensor
        :param y: Ground truth label
        :param item: tensors to be classified
        :return: predicted labels
        """
        y_predict = []
        for i in item:
            point_distances = []
            for ipt in range(X.shape[0]):
                distances = self.distance(X[ipt, :], i)
                point_distances.append(distances)

            point_distances = torch.tensor(point_distances)
            k_neighbors = torch.argsort(point_distances)[:self.k]
            y_label = y[k_neighbors]
            major_class = mode(y_label)
            major_class = major_class.mode[0]
            y_predict.append(major_class)

        return torch.tensor(y_predict)

if __name__ == '__main__':
    iris = load_iris()
    X = torch.tensor(iris.data)
    y = torch.tensor(iris.target)
    torch.manual_seed(0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    knn = KNN(k=5, X=x_train)
    y_pred = knn.fit_predict(x_train, y_train, x_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
