import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

class SVM:
    def __init__(self, X, y, C=1.0):
        self.total_samples, self.features_count = X.shape[0], X.shape[1]
        self.n_classes = len(y.unique())
        self.learning_rate = 0.001
        self.C = C

    def loss(self, X, W, y):
        """
        C parameter tells the SVM optimization how much you want to avoid misclassifying each training
        example. For large values of C, the optimization will choose a smaller-margin hyperplane if that
        hyperplane does a better job of getting all the training points classified correctly. Conversely,
        a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane,
        even if that hyperplane misclassifies more points. For very tiny values of C, you should get
        misclassified examples, often even if your training data is linearly separable.

        :param X:
        :param W:
        :param y:
        :return:
        """
        num_samples = X.shape[0]
        distances = 1 - y * (torch.mm(X, W.T))

        distances[distances < 0] = 0
        hinge_loss = self.C * (torch.sum(distances) // num_samples)
        cost = 1 / 2 * torch.mm(W, W.T) + hinge_loss
        return cost

    def gradient_update(self, W, X, y):
        """
        :param W: Weight Matrix
        :param X: Input Tensor
        :param y: Ground truth tensor
        :return: change in weight
        """
        distance = 1 - (y * torch.mm(X, W.T))
        dw = torch.zeros((1, X.shape[1]),dtype=torch.double)
        for idx, dist in enumerate(distance):
            if max(0, dist) == 0:
                di = W
            else:
                di = W - (self.C * y[idx] * X[idx])

            dw += di

        dw = dw / len(y)
        return dw

    def fit(self, X, y, max_epochs):
        """
        :param X: Input Tensor
        :param y: Output tensor
        :param max_epochs: Number of epochs the complete dataset is passed through the model
        :return: learned weight of the svm model
        """
        weight = torch.randn((1, X.shape[1]), dtype=torch.double) * torch.sqrt(torch.scalar_tensor(1./X.shape[1]))
        cost_threshold = 0.0001
        previous_cost = float('inf')
        nth = 0
        for epoch in range(1, max_epochs+1):
            X, y = shuffle(X, y)
            for idx, x in enumerate(X):
                weight_update = self.gradient_update(weight, torch.tensor(x).unsqueeze(0), y[idx])
                weight = weight - (self.learning_rate * weight_update)

            if epoch % 100 == 0:
                cost = self.loss(X, weight, y)
                print(f'Loss at epoch {epoch}: {cost}')
                if abs(previous_cost - cost) < cost_threshold * previous_cost:
                    return weight
                previous_cost = cost
                nth += 1
        return weight

if __name__ == '__main__':
    num_epochs = 1000
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    X_normalized = MinMaxScaler().fit_transform(X)
    X = torch.tensor(X_normalized)
    y = torch.tensor(breast_cancer.target).unsqueeze(1)
    bias = torch.ones((X.shape[0], 1))
    X = torch.cat((bias, X), dim=1)
    torch.manual_seed(0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svm = SVM(x_train, y_train)
    model_weights = svm.fit(x_train, y_train, max_epochs=num_epochs)
    y_pred = torch.sign(torch.mm(x_test, model_weights.T))
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
