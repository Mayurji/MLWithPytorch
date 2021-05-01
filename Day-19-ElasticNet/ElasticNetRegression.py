import torch
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

class ElasticNetRegression:
    def __init__(self, learning_rate, max_iterations, l1_penality, l2_penality):
        self.lr = learning_rate
        self.max_iterations = max_iterations
        self.l1_penality = l1_penality
        self.l2_penality = l2_penality

    def normalization(self, X):
        """
        :param X: Input tensor
        :return: Normalized input using l2 norm.
        """
        l2 = torch.norm(X, p=2, dim=-1)
        l2[l2 == 0] = 1
        return X / l2.unsqueeze(1)

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.w = torch.zeros(self.n, dtype=torch.double).unsqueeze(1)
        self.b = 0.0
        self.X = X
        self.y = y
        for i in range(self.max_iterations):
            self.update_weights()

        return self

    def update_weights(self):
        y_pred = self.predict(self.X)
        dw = torch.zeros(self.n).unsqueeze(1)
        for j in range(self.n):
            if self.w[j] > 0:
                dw[j] = ( - (2* torch.mm(self.X[:, j].unsqueeze(0), (self.y - y_pred)) + self.l1_penality + 2 * self.l2_penality * self.w[j])) / self.m
            else:
                dw[j] = (- (2 * torch.mm(self.X[:, j].unsqueeze(0), (self.y - y_pred)) - self.l1_penality + 2 * self.l2_penality * self.w[j])) / self.m

        db = -2 * torch.sum(self.y - y_pred) / self.m
        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr * db
        return self

    def predict(self, X):
        return torch.mm(X, self.w) + self.b

if __name__ == '__main__':
    data = load_boston()
    regression = ElasticNetRegression(max_iterations=1000, learning_rate=0.001, l1_penality=500, l2_penality=1)
    X, y = regression.normalization(torch.tensor(data.data, dtype=torch.double)), torch.tensor(data.target).unsqueeze(1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    regression.fit(x_train, y_train)
    Y_pred = regression.predict(x_test)
    print("Predicted values: ", torch.round(Y_pred[:3]))
    print("Real values: ", y_test[:3])
    print("Trained W: ", torch.round(regression.w[0]))
    print("Trained b: ", torch.round(regression.b))

