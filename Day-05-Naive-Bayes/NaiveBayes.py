import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class NaiveBayes:
    def __init__(self, X, y):
        """
        why e - epsilon ?
        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.

        :param X: input tensor
        :param y: target tensor
        :var total_samples: Number of Samples
        :var feature_count: Number of Features
        :var mu: mean
        :var sigma: variance
        :var e: epsilon
        :var n_classes: number of classes
        """
        self.total_samples, self.feature_count = X.shape[0], X.shape[1]
        self.mu = {}
        self.sigma = {}
        self.prior_probability_X = {}
        self.e = 1e-4
        self.n_classes = len(y.unique())

    def find_mu_and_sigma(self, X, y):
        """
        Bayes Theorem:
        P(Y|X) = P(X|Y) * P(Y) / P(X)

        :type mu: dict
        :type sigma: dict
        :type prior_probability: dict
        :describe mu: keys are class label and values are feature's mean values.
        :describe sigma: keys are class label and values are feature's variance values.
        :describe prior probability of x: It calculates the prior prabability of X for each class. P(X).
        :return:
        """
        for cls in range(self.n_classes):
            X_class = X[y==cls]
            self.mu[cls] = torch.mean(X_class, dim=0)
            self.sigma[cls] = torch.var(X_class, dim=0)
            self.prior_probability_X[cls] = X_class.shape[0] / X.shape[0]

    def gaussian_naive_bayes(self, X, mu, sigma):
        """
        :return: Multivariate normal(gaussian) distribution - Maximum Likelihood Estimation
        https://www.statlect.com/fundamentals-of-statistics/multivariate-normal-distribution-maximum-likelihood

        Log Likelihood Function = Constant - probability
        """
        constant = - self.feature_count / 2 * torch.log(2 * torch.tensor(np.pi)) - 0.5 * torch.sum(torch.log(sigma+self.e))
        probability = 0.5 * torch.sum(torch.pow(X-mu, 2) / (sigma + self.e), dim=1)
        return constant - probability

    def predict_probability(self, X):
        """
        Calculating probabilities for each sample input in X using prior probability
        and gaussian density function.

        torch.argmax: To find the class with max-probability.

        Note: We are calculate log probabilities as in Sklearn's predict_log_proba, that why we have + sign between
        prior probabilites and likelihood (class probability).

        :return:
        """
        probabilities = torch.zeros((X.shape[0], self.n_classes))
        for cls in range(self.n_classes):
            class_probability = self.gaussian_naive_bayes(X, self.mu[cls], self.sigma[cls])
            probabilities[:, cls] = class_probability + torch.log(torch.scalar_tensor(self.prior_probability_X[cls]))


        return torch.argmax(probabilities, dim=1)

if __name__ == '__main__':
    iris = load_iris()
    X = torch.tensor(iris.data)
    y = torch.tensor(iris.target)
    torch.manual_seed(0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    GNB = NaiveBayes(x_train, y_train)
    GNB.find_mu_and_sigma(x_train, y_train)
    y_pred = GNB.predict_probability(x_test)
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
