"""
Reading: https://en.wikipedia.org/wiki/Affinity_propagation
Applicable: Human Face Clustering
"""
import torch
from scipy.spatial.distance import pdist,squareform
from sklearn.datasets import load_breast_cancer

class AffinityPropagation:
    def __init__(self, similariy_matrix, max_iteration=200, num_iter=5, alpha=0.5, print_every=100):
        """
        :param similariy_matrix:
        :param max_iteration:
        :param num_iter:
        :param alpha:
        :param print_every:
        """
        self.s = similariy_matrix
        self.max_iteration = max_iteration
        self.alpha = alpha
        self.print_every = print_every
        N, N = self.s.shape
        self.r = torch.zeros((N, N))
        self.a = torch.zeros((N, N))

    def step(self):
        """
        :param r is responsiblity matrix, For each data point x_i, how well-suited is x_k as it exempler along with
        other exemplars.
        :param a is availability matrix, For appropriate is x_k as exemplers for x_i, while keeping other data points
        who keeps x_k as exemplar.
        :return:
        """
        N, N = self.s.shape
        old_r = self.r
        old_a = self.a
        a_plus_s = self.a + self.s

        first_max = torch.max(a_plus_s, dim=1)
        first_max_indices = torch.argmax(a_plus_s, dim=1)
        first_max = torch.reshape(torch.repeat_interleave(first_max.values, N), (N, N))
        a_plus_s[range(N), first_max_indices] = float('-inf')
        second_max = torch.max(a_plus_s, dim=1).values
        # responsibility Update
        r = self.s - first_max
        r[range(N), first_max_indices] = self.s[range(N), first_max_indices] - second_max[range(N)]
        r = self.alpha * old_r + (1 - self.alpha) * r
        rp = torch.maximum(r, torch.scalar_tensor(0))
        m = rp.size(0)
        rp.as_strided([m], [m + 1]).copy_(torch.diag(r))
        a = torch.reshape(torch.repeat_interleave(torch.sum(rp, dim=0), N),(N, N)).T - rp
        da = torch.diag(a)
        a = torch.minimum(a, torch.scalar_tensor(0))
        k = a.size(0)
        a.as_strided([k], [k+1]).copy_(da)
        # Availibility Update
        a = self.alpha * old_a + (1 - self.alpha) * a

        return r, a

    def solve(self):
        for i in range(self.max_iteration):
            self.r, self.a = self.step()

        e = self.r + self.a

        N, N = e.shape
        I = torch.where(torch.diag(e) > 0)[0]
        K = len(I)

        c = self.s[:, I]
        c = torch.argmax(c, dim=1)
        c[I] = torch.arange(0, K)
        idx = I[c]
        exemplar_indices = I
        exemplar_assignment = idx
        return exemplar_indices, exemplar_assignment

if __name__ == "__main__":
    """
    :param similarity_matrix
    It finds the similarity between data points.
    
    """
    data = load_breast_cancer()
    x = torch.tensor(data.data, dtype=torch.double)

    similarity_matrix = squareform(pdist(x, metric='euclidean'))
    similarity_matrix = torch.from_numpy(similarity_matrix)
    max_iteration = 3000
    affinity_prop = AffinityPropagation(similarity_matrix, max_iteration=max_iteration,
                                 alpha=0.5)
    indices, assignment = affinity_prop.solve()
