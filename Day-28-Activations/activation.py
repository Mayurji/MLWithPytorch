import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from MLP import MultiLayerPerceptron, CrossEntropy, normalization, accuracy_score, to_categorical

class Sigmoid:
    def __call__(self, X):
        return 1 / (1 + torch.exp(-X))

    def gradient(self, X):
        return self.__call__(X) * (1 - self.__call__(X))

class Softmax:
    def __call__(self, X):
        e_x = torch.exp(X - torch.max(X, dim=-1, keepdim=True).values)
        return e_x / torch.sum(e_x, dim=1, keepdim=True)

    def gradient(self, X):
        p = self.__call__(X)
        return p * (1 - p)

class TanH:
    def __call__(self, X):
        return 2 / (1 + torch.exp(-2 * X)) - 1

    def gradient(self,X):
        return 1 - torch.pow(self.__call__(X), 2)

class Relu:
    def __call__(self, X):
        return torch.where(X>0.0, X, 0.0)

    def gradient(self, X):
        return torch.where(X >=0.0, 1.0, 0.0)

class LeakyRelu:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X):
        return torch.where(X > 0.0, X, self.alpha * X)

    def gradient(self, X):
        return torch.where(X > 0.0, 1.0, self.alpha)

class ELU:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X):
        return torch.where(X>=0.0, X, self.alpha * (torch.exp(X) - 1))

    def gradient(self, X):
        return torch.where(X >= 0.0, 1.0, self.__call__(X) + self.alpha)

class SELU():
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def __call__(self, x):
        return self.scale * torch.where(x >= 0.0, x, self.alpha*(torch.exp(x)-1))

    def gradient(self, x):
        return self.scale * torch.where(x >= 0.0, 1.0, self.alpha * torch.exp(x))

class SoftPlus():
    def __call__(self, x):
        return torch.log(1 + torch.exp(x))

    def gradient(self, x):
        return 1 / (1 + torch.exp(-x))

if __name__ == '__main__':
    data = load_digits()
    X = normalization(torch.tensor(data.data, dtype=torch.double))
    y = torch.tensor(data.target)

    # Convert the nominal y values to binary
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    # MLP
    clf = MultiLayerPerceptron(n_hidden=16,
                               n_iterations=1000,
                               learning_rate=0.01, activation_function_hidden_layer=Sigmoid(),
                               activation_function_output_layer=Softmax())

    clf.fit(X_train, y_train)
    y_pred = torch.argmax(clf.predict(X_test), dim=1)
    y_test = torch.argmax(y_test, dim=1)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
