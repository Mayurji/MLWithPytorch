import torch
from sklearn.datasets import make_blobs

class LogisticRegression:
    def __init__(self, X):
        """
        :param X: Input tensor
        :keyword lr: learning rate
        :keyword epochs: number of times the model iterates over complete dataset
        :keyword weights: parameters learned during training
        :keyword bias: parameter learned during training
        """
        self.lr = 0.1
        self.epochs = 1000
        self.m, self.n = X.shape
        self.weights = torch.zeros((self.n, 1), dtype=torch.double)
        self.bias = 0

    def sigmoid(self, z):
        """
        :param z: latent variable represents (wx + b)
        :return: squashes the real value between 0 and 1 representing probability score.
        """
        return 1 / (1 + torch.exp(-z))

    def loss(self, yhat):
        """
        :param yhat: Estimated y
        :return: Log loss - When y=1, it cancels out half function, remaining half is considered for loss calculation and vice-versa
        """
        return -(1 / self.m) * torch.sum(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))

    def gradient(self, y_predict):
        """
        :param y_predict: Estimated y
        :return: gradient is calculated to find how much change is required in parameters to reduce the loss.
        """
        dw = 1 / self.m * torch.mm(X.T, (y_predict - y))
        db = 1 / self.m * torch.sum(y_predict - y)
        return dw, db

    def run(self, X, y):
        """
        :param X: Input tensor
        :param y: Output tensor
        :var y_predict: Predicted tensor
        :var cost: Difference between ground truth and predicted
        :var dw, db: Weight and bias update for weight tensor and bias scalar
        :return: updated weights and bias
        """
        for epoch in range(1, self.epochs + 1):

            y_predict = self.sigmoid(torch.mm(X, self.weights) + self.bias)
            cost = self.loss(y_predict)
            dw, db = self.gradient(y_predict)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0:
                print(f"Cost after iteration {epoch}: {cost}")

        return self.weights, self.bias

    def predict(self, X):
        """
        :param X: Input tensor
        :var y_predict_labels: Converts float value to int/bool true(1) or false(0)
        :return: outputs labels as 0 and 1
        """
        y_predict = self.sigmoid(torch.mm(X, self.weights) + self.bias)
        y_predict_labels = y_predict > 0.5

        return y_predict_labels

if __name__ == '__main__':
    """
    :var manual_seed: for reproducing the results
    :desc unsqueeze: adds a dimension to the tensor at specified position.
    """
    torch.manual_seed(0)
    X, y = make_blobs(n_samples=1000, centers=2)
    X = torch.tensor(X)
    y = torch.tensor(y).unsqueeze(1)
    lr = LogisticRegression(X)
    w, b = lr.run(X, y)
    y_predict = lr.predict(X)

    print(f"Accuracy: {torch.sum(y == y_predict) // X.shape[0]}")
