"""
Reference: https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a
Playground: https://distill.pub/2016/misread-tsne/
Wiki: https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
"""
import torch
import logging
from sklearn.datasets import load_iris, load_digits, load_diabetes
class TSNE:
    """
    The goal is to take a set of points in a high-dimensional space and find a faithful representation of those
    points in a lower-dimensional space, typically the 2D plane. The algorithm is non-linear and adapts to the
    underlying data, performing different transformations on different regions. Those differences can be a major
    source of confusion.
    """
    def __init__(self, n_components=2, preplexity=5.0, max_iter=1, learning_rate=200):
        """
        :param n_components:
        :param preplexity: how to balance attention between local and global aspects of your data. The parameter is,
        in a sense, a guess about the number of close neighbors each point has. Typical value between 5 to 50.
        With small value of preplexity, the local groups are formed and with increasing preplexity global groups are
        formed. A perplexity is more or less a target number of neighbors for our central point.
        :param max_iter: Iterations to stabilize the results and converge.
        :param learning_rate:
        """
        self.max_iter = max_iter
        self.preplexity = preplexity
        self.n_components = n_components
        self.initial_momentum = 0.5
        self.final_momentum = 0.8
        self.min_gain = 0.01
        self.lr = learning_rate
        self.tol = 1e-5
        self.preplexity_tries = 50

    def l2_distance(self, X):
        """
        :return: It helps in identifying the distance between the two distribution.
        """
        sum_X = torch.sum(X * X, dim=1)
        return (-2* torch.mm(X, X.T) + sum_X).T + sum_X

    def get_pairwise_affinities(self, X):
        """
        :param X: High dimensional input
        :return: a (Gaussian) probability distribution over pairs of high-dimensional objects in such a way that similar
        objects are assigned a higher probability while dissimilar points are assigned a lower probability. To find
        variance for this distribution we use Binary search. The variance is calculated between fixed preplexity given
        by the user.
        """
        affines = torch.zeros((self.n_samples, self.n_samples), dtype=torch.float32)
        target_entropy = torch.log(torch.scalar_tensor(self.preplexity))
        distance = self.l2_distance(X)
        for i in range(self.n_samples):
            affines[i, :] = self.binary_search(distance[i], target_entropy)

        #affines = torch.diagonal(affines).fill_(1.0e-12)
        affines[torch.eye(affines.shape[0]).byte()] = 1.0e-12
        affines = affines.clip(min=1e-100)
        affines = (affines + affines.T)/(2*self.n_samples)
        return affines

    def q_distribution(self, D):
        """
        A (Student t-distirbution)distribution is learnt in lower dimensional space, n_samples and n_components
        (2 or 3 dimension), and similar to above method 'get_pairwise_affinities', we find the probability of the
        data points with high probability for closer points and less probability for disimilar points.
        """
        Q = 1.0 / (1.0 + D)
        Q[torch.eye(Q.shape[0]).byte()] = 0.0
        Q = Q.clip(min=1e-100)
        return Q

    def binary_search(self, dist, target_entropy):
        """
        SNE performs a binary search for the value of sigma that produces probability distribution with a fixed
        perplexity that is specified by the user.
        """
        precision_minimum = 0
        precision_maximum = 1.0e15
        precision = 1.0e5

        for _ in range(self.preplexity_tries):
            denominator = torch.sum(torch.exp(-dist[dist > 0.0] / precision))
            beta = torch.exp(-dist / precision) / denominator

            g_beta = beta[beta > 0.0]
            # Shannon Entropy
            entropy = -torch.sum(g_beta * torch.log2(g_beta))
            error = entropy - target_entropy

            if error > 0:
                precision_maximum = precision
                precision = (precision + precision_minimum) / 2.0
            else:
                precision_minimum = precision
                precision = (precision + precision_maximum) / 2.0

            if torch.abs(error) < self.tol:
                break

        return beta

    def fit_transform(self, X):
        self.n_samples, self.n_features =  X.shape[0], X.shape[1]
        Y = torch.randn(self.n_samples, self.n_components)
        velocity = torch.zeros_like(Y)
        gains = torch.ones_like(Y)
        P = self.get_pairwise_affinities(X)

        iter_num = 0
        while iter_num < self.max_iter:
            iter_num += 1
            D = self.l2_distance(Y)
            Q = self.q_distribution(D)
            Q_n = Q /torch.sum(Q)

            pmul = 4.0 if iter_num < 100 else 1.0
            momentum = 0.5 if iter_num < 20 else 0.8

            grads = torch.zeros(Y.shape)
            for i in range(self.n_samples):
                """
                Optimization using gradient to converge between the true P and estimated Q distrbution.
                """
                grad = 4 * torch.mm(((pmul * P[i] - Q_n[i]) * Q[i]).unsqueeze(0), Y[i] -Y)
                grads[i] = grad

            gains = (gains + 0.2) * ((grads > 0) != (velocity > 0)) + (gains * 0.8) * ((grads > 0) == (velocity > 0))
            gains = gains.clip(min=self.min_gain)

            velocity = momentum * velocity - self.lr * (gains * grads)
            Y += velocity
            Y = Y - torch.mean(Y, 0)
            error = torch.sum(P * torch.log(P/Q_n))
            print("Iteration %s, error %s" % (iter_num, error))
        return Y

if __name__ == '__main__':
    data = load_diabetes()
    torch.manual_seed(42)
    X = torch.tensor(data.data, dtype=torch.double)
    print(max(X[1,:]))
    y = torch.tensor(data.target)
    print(y.shape)
    tsne = TSNE(n_components=2)
    tsne.fit_transform(X)
