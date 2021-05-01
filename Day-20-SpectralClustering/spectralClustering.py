import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

def SpectralClustering(X, K=8, adj=True, metric='euclidean', sim_graph='fully_connect', sigma=1, knn=10, epsilon=0.5, normalized=1):

    if not adj:
        adjacency_matrix = squareform(X, metric=metric)
    else:
        adjacency_matrix = X

    if sim_graph == 'fully_connect':
        adjacency_matrix = torch.from_numpy(adjacency_matrix)
        w = torch.exp(-adjacency_matrix/ (2 * sigma))
    elif sim_graph =='eps_neighbor':
        adjacency_matrix = torch.from_numpy(adjacency_matrix)
        w = (adjacency_matrix <= epsilon).type(torch.DoubleTensor)
    elif sim_graph == 'knn':
        adjacency_matrix = torch.from_numpy(adjacency_matrix)
        w = torch.zeros(adjacency_matrix.shape)
        adjacency_sort = torch.argsort(adjacency_matrix, dim=1)
        for i in range(adjacency_sort.shape[0]):
            w[i, adjacency_sort[i, :][:(knn+1)]] = 1
    elif sim_graph == 'mutual_knn':
        adjacency_matrix = torch.from_numpy(adjacency_matrix)
        w1 = torch.zeros(adjacency_matrix.shape)
        adjacency_sort = torch.argsort(adjacency_matrix, dim=1)
        for i in range(adjacency_matrix.shape[0]):
            for j in adjacency_sort[i, :][:(knn+1)]:
                if i==j:
                    w1[i, i] = 1
                elif w1[i, j] == 0 and w1[j, i]==0:
                    w1[i, j] = 0.5
                else:
                    w1[i, j] = w1[j, i] = 1
        w = w1[w1>0.5].type(torch.DoubleTensor).clone
    else:
        raise ValueError("The 'sim_graph' argument should be one of the strings, 'fully_connect', 'eps_neighbor', 'knn', or 'mutual_knn'!")

    D = torch.diag(torch.sum(w, dim=1))
    L = D - w

    if normalized == 1:
        D_INV = torch.diag(1/torch.diag(D))
        lambdas, V = torch.eig(torch.mm(D_INV, L), eigenvectors=True)
        ind = torch.argsort(torch.norm(torch.reshape(lambdas[:,0], (1, len(lambdas))), dim=0))
        V_K = V[:, ind[:K]]

    elif normalized == 2:
        D_INV_SQRT = torch.diag(1/torch.sqrt(torch.diag(D)))
        lambdas, V = torch.eig(torch.matmul(torch.matmul(D_INV_SQRT, L), D_INV_SQRT))
        ind = torch.argsort(torch.norm(torch.reshape(lambdas[:,0], (1, len(lambdas))), dim=0))
        V_K = torch.real(V[:, ind[:,K]])
        if any(V_K.sum(dim=1) == 0):
            raise ValueError("Can't normalize the matrix with the first K eigenvectors as columns! Perhaps the number of clusters K or the number of neighbors in k-NN is too small.")
        V_K = V_K/torch.reshape(torch.norm(V_K, dim=1), (V_K.shape[0], 1))
    else:
        lambdas, V = torch.eig(L)
        ind = torch.argsort(torch.norm(torch.reshape(lambdas[:,0], (1, len(lambdas))), dim=0))
        V_K = torch.real(V[:, ind[:K]])

    kmeans = KMeans(n_clusters=K, init='k-means++', random_state=0).fit(V_K)
    return kmeans

if __name__ == '__main__':
    moon_data, moon_labels = make_moons(1000, noise=0.05)
    moon_data = torch.tensor(moon_data)
    moon_labels = torch.tensor(moon_labels)
    # Compute the adjacency matrix
    Adj_mat = squareform(pdist(moon_data, metric='euclidean', p=2))
    # Spectral clustering...
    spec_re1 = SpectralClustering(Adj_mat, K=2, sim_graph='fully_connect', sigma=0.01, normalized=1)
    spec_re2 = SpectralClustering(Adj_mat, K=2, sim_graph='knn', knn=10, normalized=1)

    # Often need to change figsize when doing subplots
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x=moon_data[:, 0], y=moon_data[:, 1], c=spec_re1.labels_, s=2)
    plt.colorbar()
    plt.title('Fully connected graph with RBF kernel ($\sigma=0.01$)')

    plt.subplot(1, 2, 2)
    plt.scatter(x=moon_data[:, 0], y=moon_data[:, 1], c=spec_re2.labels_, s=2)
    plt.colorbar()
    plt.title('$k$-Nearest Neighbor graphs ($k=10$)')

    plt.suptitle('Spectral Clustering', y=-0.01)

    # Automatrically adjust padding between subpots
    plt.tight_layout()
    plt.show()



