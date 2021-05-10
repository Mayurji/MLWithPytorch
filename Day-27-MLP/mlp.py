"""
Multi-Layer Preceptron
"""
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

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

def accuracy_score(y, p):
    accuracy = torch.sum(y == p, dim=0) / len(y)
    return accuracy

def to_categorical(X, n_col=None):
    if not n_col:
        n_col = torch.amax(X) + 1

    one_hot = torch.zeros((X.shape[0], n_col))
    one_hot[torch.arange(X.shape[0]), X] = 1
    return one_hot

def normalization(X):
    """
    :param X: Input tensor
    :return: Normalized input using l2 norm.
    """
    l2 = torch.norm(X, p=2, dim=-1)
    l2[l2 == 0] = 1
    return X / l2.unsqueeze(1)

class CrossEntropy:
    def __init__(self):
        pass
    def loss(self, y, p):
        p = torch.clip(p, 1e-15, 1-1e-15)
        return - y * torch.log(p) - (1 -y) * torch.log(1 - p)

    def accuracy_score(self, y, p):
        return accuracy_score(torch.argmax(y, dim=1), torch.argmax(p, dim=1))

    def gradient(self, y, p):
        p = torch.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 -p)

class MultiLayerPerceptron:
    def __init__(self, n_hidden, n_iterations=1000, learning_rate=0.001):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()

    def initalize_weight(self, X, y):
        n_samples, n_features = X.shape
        _, n_outputs = y.shape
        limit = 1 / torch.sqrt(torch.scalar_tensor(n_features))
        self.W = torch.DoubleTensor(n_features, self.n_hidden).uniform_(-limit, limit)

        self.W0 = torch.zeros((1, self.n_hidden))
        limit = 1 / torch.sqrt(torch.scalar_tensor(self.n_hidden))
        self.V = torch.DoubleTensor(self.n_hidden, n_outputs).uniform_(-limit, limit)
        self.V0 = torch.zeros((1, n_outputs))

    def fit(self, X, y):
        self.initalize_weight(X, y)
        for i in range(self.n_iterations):
            hidden_input =  torch.mm(X, self.W) + self.W0
            hidden_output = self.hidden_activation(hidden_input)

            output_layer_input = torch.mm(hidden_output, self.V) + self.V0
            y_pred  = self.output_activation(output_layer_input)

            grad_wrt_first_output = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_layer_input)
            grad_v = torch.mm(hidden_output.T, grad_wrt_first_output)
            grad_v0 = torch.sum(grad_wrt_first_output, dim=0, keepdim=True)

            grad_wrt_first_hidden = torch.mm(grad_wrt_first_output, self.V.T) * self.hidden_activation.gradient(hidden_input)
            grad_w = torch.mm(X.T, grad_wrt_first_hidden)
            grad_w0 = torch.sum(grad_wrt_first_hidden, dim=0, keepdim=True)

            # Update weights (by gradient descent)
            # Move against the gradient to minimize loss
            self.V -= self.learning_rate * grad_v
            self.V0 -= self.learning_rate * grad_v0
            self.W -= self.learning_rate * grad_w
            self.W0 -= self.learning_rate * grad_w0

            # Use the trained model to predict labels of X

    def predict(self, X):
        # Forward pass:
        hidden_input = torch.mm(X,self.W) + self.W0
        hidden_output = self.hidden_activation(hidden_input)
        output_layer_input = torch.mm(hidden_output, self.V) + self.V0
        y_pred = self.output_activation(output_layer_input)
        return y_pred


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
                               learning_rate=0.01)

    clf.fit(X_train, y_train)
    y_pred = torch.argmax(clf.predict(X_test), dim=1)
    y_test = torch.argmax(y_test, dim=1)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


