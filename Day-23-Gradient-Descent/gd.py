import torch
from torch import nn
from sklearn.datasets import load_boston

class GradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=100):
        self.lr = learning_rate
        self.max_iterations = max_iterations

    def normalization(self, X):
        """
        :param X: Input tensor
        :return: Normalized input using l2 norm.
        """
        l2 = torch.norm(X, p=2, dim=-1)
        l2[l2 == 0] = 1
        return X / l2.unsqueeze(1)

    def compute_error(self, b, m, X, y):
        total_error = 0
        for i in range(0, X.shape[0]):
            total_error += (y - (torch.mm(m , X.T)) + b) ** 2
        return total_error / float(X.shape[0])

    def step(self, b_curr, m_curr, X, y, learning_rate):
        b_gradient = 0
        m_gradient = 0
        N = float(X.shape[0])
        for i in range(X.shape[0]):
            b_gradient += -(2/N) * torch.sum(y - (torch.mm(X, m_curr.T) + b_curr), dim=0)
            m_gradient += -(2/N) * torch.sum(torch.mm(X.T,  (y - (torch.mm(X, m_curr.T) + b_curr))), dim=0)

        new_b = b_curr - (learning_rate * b_gradient)
        new_m = m_curr - (learning_rate * m_gradient)
        return [new_b, new_m]

    def gradient_descent(self, X, y, start_b, start_m):
        b = start_b
        m = start_m
        for i in range(self.max_iterations):
            b, m = self.step(b_curr=b, m_curr=m, X=X, y=y, learning_rate=self.lr)

        return b, m

if __name__ == '__main__':
    data = load_boston()
    X = torch.tensor(data.data)
    y = torch.tensor(data.target).unsqueeze(1)
    initial_b = 0.0
    initial_m = torch.zeros((X.shape[1], 1), dtype=torch.double).T
    nn.init.normal(initial_m)
    gd = GradientDescent(learning_rate=0.0001,max_iterations=100)
    gd.compute_error(X=gd.normalization(X), y=y, b=initial_b, m=initial_m)
    bias, slope = gd.gradient_descent(gd.normalization(X), y, start_b=initial_b, start_m=initial_m)
    X = gd.normalization(X)
    print('y: ', y[0].item())
    print('y_pred: ', (torch.mm(slope, X[0].unsqueeze(0).T)+bias).item())

