import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
class Normalization:
    def __init__(self, X):
        self.X = X

    def z_score(self):
        mean = torch.mean(self.X, dim=0)
        return self.X.subtract(mean)/ torch.std(self.X, dim=0)

    def min_max(self):
        min = torch.min(self.X, dim=0)
        max = torch.max(self.X, dim=0)
        return self.X.subtract(min.values) / (max.values - min.values)

    def log_scaling(self):
        return torch.log(self.X)

    def clipping(self, max, min):
        if self. X > max:
            mask = self. X > max
            self.X = self.X * mask

        if self. X < min:
            mask = self. X < min
            self.X = self.X * mask

        return self.X

if __name__ == '__main__':
    data = load_iris()
    X = torch.tensor(data.data)
    y = torch.tensor(data.target).unsqueeze(1)
    cls = KNeighborsClassifier()
    normalizer = Normalization(X)
    X_transform = normalizer.z_score()
    cls.fit(X, y)
    y_pred = cls.predict(X)
    print('Without Normalization',accuracy_score(y, y_pred))
    cls.fit(X_transform, y)
    y_pred = cls.predict(X_transform)
    print('Z-Score Normalization' ,accuracy_score(y, y_pred))
    X_transform = normalizer.min_max()
    cls.fit(X_transform, y)
    y_pred = cls.predict(X_transform)
    print('Min-Max Normalization' ,accuracy_score(y, y_pred))
    X_transform = normalizer.log_scaling()
    cls.fit(X_transform, y)
    y_pred = cls.predict(X_transform)
    print('Log Scaling', accuracy_score(y, y_pred))
