"""
Checkout the below url to understand, how Bayesian regression differs from Linear Regression
https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7
https://dzone.com/articles/bayesian-learning-for-machine-learning-part-ii-lin
"""
import pandas as pd
import torch
from scipy.stats import chi2, multivariate_normal
from sklearn.model_selection import train_test_split
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = torch.mean(torch.pow(y_true - y_pred, 2))
    return mse

def polynomial_features(X, degree):
    """
    It creates polynomial features from existing set of features. For instance,
    X_1, X_2, X_3 are available features, then polynomial features takes combinations of
    these features to create new feature by doing X_1*X_2, X_1*X_3, X_2*X3.

    For Degree 2:
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


class BayesianRegression:
    def __init__(self, n_draws, mu_0, omega_0, nu_0, sigma_sq_0, polynomial_degree=0, credible_interval=95):
        """
        Bayesian regression model. If poly_degree is specified the features will
        be transformed to with a polynomial basis function, which allows for polynomial
        regression. Assumes Normal prior and likelihood for the weights and scaled inverse
        chi-squared prior and likelihood for the variance of the weights.

        :param n_draws:  The number of simulated draws from the posterior of the parameters.
        :param mu_0:  The mean values of the prior Normal distribution of the parameters.
        :param omega_0: The precision matrix of the prior Normal distribution of the parameters.
        :param nu_0: The degrees of freedom of the prior scaled inverse chi squared distribution.
        :param sigma_sq_0: The scale parameter of the prior scaled inverse chi squared distribution.
        :param polynomial_degree: The polynomial degree that the features should be transformed to. Allows
        for polynomial regression.
        :param credible_interval: The credible interval (ETI in this impl.). 95 => 95% credible interval of the posterior
        of the parameters.
        """
        self.n_draws = n_draws
        self.polynomial_degree = polynomial_degree
        self.credible_interval = credible_interval

        # Prior parameters
        self.mu_0 = mu_0
        self.omega_0 = omega_0
        self.nu_0 = nu_0
        self.sigma_sq_0 = sigma_sq_0

    def scaled_inverse_chi_square(self, n, df, scale):
        """
        Allows for simulation from the scaled inverse chi squared
        distribution. Assumes the variance is distributed according to
        this distribution.
        :param n:
        :param df:
        :param scale:
        :return:
        """
        X = chi2.rvs(size=n, df=df)
        sigma_sq = df * scale / X
        return sigma_sq

    def fit(self, X, y):
        # For polynomial transformation
        if self.polynomial_degree:
            X = polynomial_features(X, degree=self.polynomial_degree)

        n_samples, n_features = X.shape[0], X.shape[1]
        X_X_T = torch.mm(X.T, X)

        # Least squares approximate of beta
        beta_hat = torch.mm(torch.mm(torch.pinverse(X_X_T), X.T), y)

        # The posterior parameters can be determined analytically since we assume
        # conjugate priors for the likelihoods.
        # Normal prior / likelihood => Normal posterior
        mu_n = torch.mm(torch.pinverse(X_X_T + self.omega_0), torch.mm(X_X_T, beta_hat) + torch.mm(self.omega_0, self.mu_0.unsqueeze(1)))
        omega_n = X_X_T + self.omega_0
        nu_n = self.nu_0 + n_samples

        # Scaled inverse chi-squared prior / likelihood => Scaled inverse chi-squared posterior
        sigma_sq_n = (1.0/nu_n) * (self.nu_0 * self.sigma_sq_0 + torch.mm(y.T, y) + torch.mm(torch.mm(self.mu_0.unsqueeze(1).T, self.omega_0), self.mu_0.unsqueeze(1)) - torch.mm(mu_n.T, torch.mm(omega_n, mu_n)))

        # Simulate parameter values for n_draws
        beta_draws = torch.empty((self.n_draws, n_features))
        for i in range(self.n_draws):
            sigma_sq = self.scaled_inverse_chi_square(n=1, df=nu_n, scale=sigma_sq_n)
            beta = multivariate_normal.rvs(size=1, mean=mu_n[:,0], cov=sigma_sq * torch.pinverse(omega_n))
            beta_draws[1, :] = torch.tensor(beta,dtype=torch.float)

        # Select the mean of the simulated variables as the ones used to make predictions
        self.w = torch.mean(beta_draws, dim=0, dtype=torch.double)

        # Lower and upper boundary of the credible interval
        l_eti = 0.50 - self.credible_interval / 2
        u_eti = 0.50 + self.credible_interval / 2
        self.eti = torch.tensor([[torch.quantile(beta_draws[:, i], q=l_eti), torch.quantile(beta_draws[:, i], q=u_eti)] for i in range(n_features)], dtype=torch.double)

    def predict(self, X, eti=False):
        if self.polynomial_degree:
            X = polynomial_features(X, degree=self.polynomial_degree)
        y_pred = torch.mm(X, self.w.unsqueeze(1))
        # If the lower and upper boundaries for the 95%
        # equal tail interval should be returned
        if eti:
            lower_w = self.eti[:, 0]
            upper_w = self.eti[:, 1]

            y_lower_prediction = torch.mm(X, lower_w.unsqueeze(1))
            y_upper_prediction = torch.mm(X, upper_w.unsqueeze(1))

            return y_pred, y_lower_prediction, y_upper_prediction

        return y_pred

if __name__ == '__main__':
    data = pd.read_csv('temp.txt', sep="\t")
    X = torch.tensor(data["time"].values).unsqueeze(0).T
    y = torch.tensor(data["temp"].values).unsqueeze(0).T
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    n_samples, n_features = X.shape[0], X.shape[1]
    mu_0 = torch.zeros(n_features, dtype=torch.double)
    omega_0 = torch.diag(torch.tensor([0.0001] * n_features, dtype=torch.double))
    nu_0 = 1
    sigma_sq_0 = 100
    credible_interval = 0.40
    classifier = BayesianRegression(n_draws=2000,
                                    polynomial_degree=4,
                                    mu_0=mu_0,
                                    omega_0=omega_0,
                                    nu_0=nu_0,
                                    sigma_sq_0=sigma_sq_0,
                                    credible_interval=credible_interval)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    y_pred_, y_lower_, y_upper_ = classifier.predict(X=X, eti=True)
    print("Mean Squared Error:", mse)
    #
    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results
    m1 = plt.scatter(366 * x_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * x_test, y_test, color=cmap(0.5), s=10)
    p1 = plt.plot(366 * X, y_pred_, color="black", linewidth=2, label="Prediction")
    p2 = plt.plot(366 * X, y_lower_, color="gray", linewidth=2, label="{0}% Credible Interval".format(credible_interval))
    p3 = plt.plot(366 * X, y_upper_, color="gray", linewidth=2)
    plt.axis((0, 366, -20, 25))
    plt.suptitle("Bayesian Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend(loc='lower right')
    # plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.legend(loc='lower right')

    plt.show()
