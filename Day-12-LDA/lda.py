import torch
from NaiveBayes import NaiveBayes
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LDA:
    def __init__(self):
        self.w = None

    def covariance_matrix(self, X):
        """
        :param X: Input tensor
        :return: cavariance of input tensor
        """
        centering_X = X - torch.mean(X, dim=0)
        cov = torch.mm(centering_X.T, centering_X) / (centering_X.shape[0] - 1)
        return cov

    def fit(self, X, y):
        """
        :param X: Input tensor
        :param y: output tensor
        :return: transformation vector - to convert high dimensional input space into lower dimensional
        subspace.
        X1, X2 are samples based on class. cov_1 and cov_2 measures how features of samples of each class are related.

        """
        X1 = X[y==0]
        X2 = X[y==1]
        cov_1 = self.covariance_matrix(X1)
        cov_2 = self.covariance_matrix(X2)
        cov_total = cov_1 + cov_2
        mean1 = torch.mean(X1, dim=0)
        mean2 = torch.mean(X2, dim=0)
        mean_diff = mean1 - mean2

        # Determine the vector which when X is projected onto it best separates the
        # data by class. w = (mean1 - mean2) / (cov1 + cov2)
        self.w = torch.mm(torch.pinverse(cov_total), mean_diff.unsqueeze(1))

    def transform(self, X, y):
        self.fit(X, y)
        X_transformed = torch.mm(X, self.w)
        return X_transformed

    def predict(self, X):
        y_pred = []
        for sample in X:
            h = torch.mm(sample.unsqueeze(0), self.w)
            y = 1 * (h < 0)
            y_pred.append(y)

        return y_pred

if __name__ == '__main__':
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    X_normalized = MinMaxScaler().fit_transform(X)
    X = torch.tensor(X_normalized)
    y = torch.tensor(breast_cancer.target)#.unsqueeze(1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lda = LDA()
    X_transformed = lda.transform(x_train, y_train)
    GNB = NaiveBayes(X_transformed, y_train)
    GNB.find_mu_and_sigma(X_transformed, y_train)
    X_test_transformed = lda.transform(x_test, y_test)
    y_pred = GNB.predict_probability(X_test_transformed)
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
