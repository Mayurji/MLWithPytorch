"""
Reference: https://github.com/eriklindernoren/ML-From-Scratch
This github repository implements high quality code as we see in official libraries like sklearn etc.
Great reference to kickstart your journey for ML programming.
"""
import torch
from sklearn.datasets import load_boston
from itertools import combinations_with_replacement
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt

class LassoRegularization:
    def __init__(self, alpha):
        """
        :param alpha:
        * When 0, the lasso regression turns into Linear Regression
        * When increases towards infinity, it turns features coefficients into zero.
        * Try out different value to find out optimized values.
        """
        self.alpha = alpha

    def __call__(self, w):
        """
        :param w: Weight vector
        :return: Penalization value for MSE
        """
        return self.alpha * torch.norm(w, p=1)

    def grad(self, w):
        """
        :param w: weight vector
        :return: weight update based on sign value, it helps in removing coefficients from W vector
        torch.sign:
        a
        tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
        torch.sign(a)
        tensor([ 1., -1.,  0.,  1.])
        """
        return self.alpha * torch.sign(w)

class RidgeRegularization:
    def __init__(self, alpha):
        """
        :param alpha:
        * When 0, the lasso regression turns into Linear Regression
        * When increases towards infinity, it turns features coefficients into zero.
        * Try out different value to find out optimized values.
        """
        self.alpha = alpha

    def __call__(self, w):
        """
        :param w: Weight vector
        :return: Penalization value for MSE
        """
        return self.alpha * 0.5 * torch.mm(w.T, w)

    def grad(self, w):
        """
        :param w: weight vector
        :return: weight update based on sign value, it helps in reducing the coefficient values from W vector
        torch.sign:
        a
        tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
        torch.sign(a)
        tensor([ 1., -1.,  0.,  1.])
        """
        return self.alpha * w

class Regression:
    def __init__(self, learning_rate, epochs, regression_type='lasso'):
        """
        :param learning_rate: constant step while updating weight
        :param epochs: Number of epochs the data is passed through the model
        Initalizing regularizer for Lasso Regression.
        """
        self.lr = learning_rate
        self.epochs = epochs
        if regression_type == 'lasso':
            self.regularization = LassoRegularization(alpha=1.0)
        else:
            self.regularization = RidgeRegularization(alpha=2.0)

    def normalization(self, X):
        """
        :param X: Input tensor
        :return: Normalized input using l2 norm.
        """
        l2 = torch.norm(X, p=2, dim=-1)
        l2[l2 == 0] = 1
        return X / l2.unsqueeze(1)

    def polynomial_features(self, X, degree):
        """
        It creates polynomial features from existing set of features. For instance,
        X_1, X_2, X_3 are available features, then polynomial features takes combinations of
        these features to create new feature by doing X_1*X_2, X_1*X_3, X_2*X3.

        combinations output: [(), (0,), (1,), (2,), (3,), (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
        :param X: Input tensor (For Iris Dataset, (150, 4))
        :param degree: Polynomial degree of 2, i.e we'll have product of two feature vector at max.
        :return: Output tensor (After adding polynomial features, the number of features increases to 15)
        """
        n_samples, n_features = X.shape[0], X.shape[1]
        def index_combination():
            combinations = [combinations_with_replacement(range(n_features), i) for i in range(0, degree+1)]
            flat_combinations = [item for sublists in combinations for item in sublists]
            return flat_combinations

        combinations = index_combination()
        n_output_features = len(combinations)
        X_new = torch.empty((n_samples, n_output_features))

        for i, index_combs in enumerate(combinations):
            X_new[:, i] = torch.prod(X[:, index_combs], dim=1)

        X_new = X_new.type(torch.DoubleTensor)
        return X_new

    def weight_initialization(self, n_features):
        """
        :param n_features: Number of features in the data
        :return: creating weight vector using uniform distribution.
        """
        limit = 1 / torch.sqrt(torch.scalar_tensor(n_features))
        #self.w = torch.FloatTensor((n_features,)).uniform(-limit, limit)
        self.w = torch.distributions.uniform.Uniform(-limit, limit).sample((n_features, 1))
        self.w = self.w.type(torch.DoubleTensor)

    def fit(self, X, y):
        """
        :param X: Input tensor
        :param y: ground truth tensor
        :return: updated weight vector for prediction
        """
        self.training_error = {}
        self.weight_initialization(n_features=X.shape[1])
        for epoch in range(1, self.epochs+1):
            y_pred = torch.mm(X, self.w)
            mse = torch.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_error[epoch] = mse.item()
            grad_w = torch.mm(-(y - y_pred).T, X).T + self.regularization.grad(self.w)
            self.w -= self.lr * grad_w


    def predict(self, X):
        """
        :param X: input tensor
        :return: predicted output using learned weight vector
        """
        y_pred = torch.mm(X, self.w)
        return y_pred

if __name__ == '__main__':
    boston = load_boston()
    torch.manual_seed(0)
    X = torch.tensor(boston.data, dtype=torch.double)
    y = torch.tensor(boston.target, dtype=torch.double).unsqueeze(1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    regression = Regression(learning_rate=0.0001, epochs=3000, regression_type='lasso')
    regression.fit(regression.normalization(regression.polynomial_features(x_train, degree=1)), y_train)
    y_pred = regression.predict(regression.normalization(regression.polynomial_features(x_test, degree=1)))
    plt.figure(figsize=(6, 6))
    sb.scatterplot(list(regression.training_error.keys()), list(regression.training_error.values()))
    plt.show()
