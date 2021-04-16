"""
Blog post GMM: https://brilliant.org/wiki/gaussian-mixture-model/
"""
import torch
import math
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class GMM:
    def __init__(self, k, max_epochs=100, tolerance=1e-8):
        """
        :param k: the number of clusters the algorithm will form.
        :param max_epochs: The number of iterations the algorithm will run for if it does
        not converge before that.
        :param tolerance: float
        If the difference of the results from one iteration to the next is
        smaller than this value we will say that the algorithm has converged.
        """
        self.k = k
        self.parameters = []
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.responsibility = None
        self.responsibilities = []
        self.sample_assignments = None

    def normalization(self, X):
        """
        :param X: Input tensor
        :return: Normalized input using l2 norm.
        """
        l2 = torch.norm(X, p=2, dim=-1)
        l2[l2 == 0] = 1
        return X / l2.unsqueeze(1)

    def covariance_matrix(self, X):
        """
        :param X: Input tensor
        :return: cavariance of input tensor
        """
        centering_X = X - torch.mean(X, dim=0)
        cov = torch.mm(centering_X.T, centering_X) / (centering_X.shape[0] - 1)
        return cov

    def random_gaussian_initialization(self, X):
        """
        Since we are using iris dataset, we know the no. of class is 3.
        We create three gaussian distribution representing each class with
        random sampling of data to find parameters like μ and 𝚺/N (covariance matrix)
        for each class
        :param X: input tensor
        :return: 3 randomly selected mean and covariance of X, each act as a separate cluster
        """
        n_samples = X.shape[0]
        self.prior = (1 / self.k) * torch.ones(self.k)
        for cls in range(self.k):
            parameter = {}
            parameter['mean'] = X[torch.randperm(n_samples)[:1]]
            parameter['cov'] = self.covariance_matrix(X)
            self.parameters.append(parameter)

    def multivariate_gaussian_distribution(self, X, parameters):
        """
        Checkout the equation from Multi-Dimensional Model from blog link posted above.
        We find the likelihood of each sample w.r.t to the parameters initialized above for each separate cluster.
        :param X: Input tensor
        :param parameters: mean, cov of the randomly initialized gaussian
        :return: Likelihood of each sample belonging to a cluster with random initialization of mean and cov.
        Since it is a multivariate problem we have covariance and not variance.
        """
        n_features = X.shape[1]
        mean = parameters['mean']
        cov = parameters['cov']
        determinant = torch.det(cov)
        likelihoods = torch.zeros(X.shape[0])
        for i, sample in enumerate(X):
            dim = torch.scalar_tensor(n_features, dtype=torch.float)
            coefficients = 1.0/ torch.sqrt(torch.pow((2.0 * math.pi), dim) * determinant)
            exponent = torch.exp( -0.5 * torch.mm(torch.mm((sample - mean) ,torch.pinverse(cov)) , (sample - mean).T))
            likelihoods[i] = coefficients * exponent

        return likelihoods

    def get_likelihood(self, X):
        """
        Previously, we have initialized 3 different mean and covariance in random_gaussian_initialization(). Now around
        each of these mean and cov, we see likelihood of the each sample using multivariate gaussian distribution.
        :param X:
        :return: Storing the likelihood of each sample belonging to a cluster with random initialization of mean and cov.
        Since it is a multivariate problem we have covariance and not variance.
        """
        n_samples = X.shape[0]
        likelihoods_cls = torch.zeros((n_samples, self.k))
        for cls in range(self.k):
            likelihoods_cls[:, cls] = self.multivariate_gaussian_distribution(X, self.parameters[cls])

        return likelihoods_cls

    def expectation(self, X):
        """
        Expectation Maximization Algorithm is used to find the optimized value of randomly initialized mean and cov.
        Expectation refers to probability. Here, It calculates the probabilities of X belonging to different cluster.
        :param X: input tensor
        :return: Max probability of each sample belonging to a particular class.
        """
        weighted_likelihood = self.get_likelihood(X) * self.prior
        sum_likelihood =  torch.sum(weighted_likelihood, dim=1).unsqueeze(1)
        # Determine responsibility as P(X|y)*P(y)/P(X)
        # responsibility stores each sample's probability score corresponding to each class
        self.responsibility = weighted_likelihood /sum_likelihood
        # Assign samples to cluster that has largest probability
        self.sample_assignments = self.responsibility.argmax(dim=1)
        # Save value for convergence check
        self.responsibilities.append(torch.max(self.responsibility, dim=1))

    def maximization(self, X):
        """
        Iterate through clusters and updating mean and covariance.
        Finding updated mean and covariance using probability score of each sample w.r.t each class
        :param X:
        :return: Updated mean, covariance and priors
        """
        for i in range(self.k):
            resp = self.responsibility[:, i].unsqueeze(1)
            mean = torch.sum(resp * X, dim=0) / torch.sum(resp)
            covariance = torch.mm((X - mean).T, (X - mean) * resp) / resp.sum()
            self.parameters[i]['mean'], self.parameters[i]['cov'] = mean.unsqueeze(0), covariance

        n_samples = X.shape[0]
        self.prior = self.responsibility.sum(dim=0) / n_samples

    def convergence(self, X):
        """Convergence if || likehood - last_likelihood || < tolerance """
        if len(self.responsibilities) < 2:
            return False
        difference = torch.norm(self.responsibilities[-1].values - self.responsibilities[-2].values)
        return difference <= self.tolerance

    def predict(self, X):
        self.random_gaussian_initialization(X)

        for _ in range(self.max_epochs):
            self.expectation(X)
            self.maximization(X)
            break

            if self.convergence(X):
                break

        self.expectation(X)
        return self.sample_assignments

if __name__ == '__main__':
    iris = load_iris()
    torch.manual_seed(0)
    X = torch.tensor(iris.data, dtype=torch.float)
    y = torch.tensor(iris.target)
    n_classes = len(torch.unique(y))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    gmm = GMM(k=n_classes, max_epochs=2000)
    y_pred = gmm.predict(x_train)
    print(f'Accuracy Score: {accuracy_score(y_train, y_pred)}')

