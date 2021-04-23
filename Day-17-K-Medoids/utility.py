import torch
from sklearn.metrics import accuracy_score
import numpy as np

class SquareLoss:
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        return 0.5 * torch.pow((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class CrossEntropy:
    def __init__(self):
        pass

    def loss(self, y, p):
        p = torch.clip(p, 1e-15, 1 - 1e-15)
        return - y * torch.log(p) - (1 - y) * torch.log(1 - p)

    def accuracy(self, y, p):
        return accuracy_score(torch.argmax(y, dim=1), torch.argmax(p, dim=1))

    def gradient(self, y, p):
        p = torch.clip(p, 1e-15, 1 - 1e-15)
        return -(y/p) + (1-y) / (1-p)

def euclidean_distance(x1, x2):
    """
    :param x1: input tensor
    :param x2: input tensor
    :return: distance between tensors
    """

    return torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0))

def to_categorical(X, n_columns=None):
    if not n_columns:
        n_columns = torch.amax(X) + 1
    one_hot = torch.zeros((X.shape[0], n_columns))
    one_hot[torch.arange(X.shape[0])] = 1
    return one_hot

def mean_squared_error(y_true, y_pred):
    mse = torch.mean(torch.pow(y_true - y_pred, 2))
    return mse

def divide_on_feature(X, feature_i, threshold):

    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold


    X_1 = torch.tensor([sample.numpy() for sample in X if split_func(sample)])
    X_2 = torch.tensor([sample.numpy() for sample in X if not split_func(sample)])

    return np.array([X_1.numpy(), X_2.numpy()], dtype='object')

def calculate_variance(X):
    mean = torch.ones(X.shape) * torch.mean(X, dim=0)
    n_samples = X.shape[0]
    variance = (1/ n_samples) * torch.diag(torch.mm((X-mean).T, (X-mean)))
    return variance
