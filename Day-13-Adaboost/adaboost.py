"""
Adaboost Algorithm Blog post:
https://www.mygreatlearning.com/blog/adaboost-algorithm/
"""
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class stump:
    "Each Stump is a weak classifier and combination of them are referred as Boosting Mechanism"
    def __init__(self):
        """
        * Polarity is used to classify sample as either 1 or -1
        * feature index is for identifying node for separating classes
        * features are compared against threshold value
        * Alpha value indicates the classifier accuracy
        """
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

class Adaboost:
    def __init__(self, num_classifiers):
        """
        :param num_classifiers: Number of weak classifiers
        """
        self.num_classifiers = num_classifiers

    def fit(self, X, y):
        """
        :param X: Input tensor
        :param y: output tensor
        :return: Creates a list of weak classifier with set of properties as
        mentioned in stump class.
        * Initialize weights to 1/N, N is number of samples
        * Iterate through different weak classifiers
        * Minimum error given for using a certain feature value threshold for predicting sample label
        * Iterate through each feature and its unique values to find the threshold value
        * Label samples with value less than threshold as -1
        * Error, Sum of weights of misclassified samples
        * If the error is over 50% we flip the polarity so that samples that were classified as 0 are
        classified as 1, and vice versa. E.g error = 0.8 => (1 - error) = 0.2
        * If this threshold resulted in the smallest error we save the configuration
        * Calculate the alpha which is used to update the sample weights,
        Alpha is also an approximation of this classifier's proficiency
        * set all predictions to '1' initially
        * The indexes where the sample values are below threshold, label them as -1
        * Updated weights and normalize to one
        * save each weak classifier
        """
        n_samples, n_features = X.shape[0], X.shape[1]
        weight = torch.zeros(n_samples).fill_(1/n_samples)
        self.clfs = []
        for _ in range(self.num_classifiers):
            clf = stump()
            minimum_error = float('inf')
            for feature_i in range(n_features):
                feature_values = X[:, feature_i].unsqueeze(1)
                unqiue_values =  feature_values.unique()
                for threshold in unqiue_values:
                     p = 1
                     prediction = torch.ones(y.shape)
                     prediction[X[:, feature_i] < threshold] = -1
                     error = torch.sum(weight[y != prediction])
                     if error > 0.5:
                         error = 1 - error
                         p = -1

                     if error < minimum_error:
                         clf.polarity = p
                         clf.threshold = threshold
                         clf.feature_index = feature_i
                         minimum_error = error

            clf.alpha = 0.5 * torch.log(1.0 - minimum_error) / (minimum_error + 1e-10)
            predictions = torch.ones(y.shape)
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1

            weight *= torch.exp(-clf.alpha * y * predictions)
            weight /= torch.sum(weight)

            self.clfs.append(clf)

    def predict(self, X):
        """
        same process as mentioned above.
        :param X:
        :return: predicted estimate of ground truth.
        """
        n_samples = X.shape[0]
        y_pred = torch.zeros((n_samples, 1))
        for clf in self.clfs:
            predictions = torch.ones(y_pred.shape)
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            y_pred += clf.alpha * predictions

        print(y_pred)
        y_pred = torch.sign(y_pred).flatten()
        print(y_pred)
        return y_pred

if __name__ == '__main__':
    breast_cancer = load_breast_cancer()
    torch.manual_seed(0)
    X = torch.tensor(breast_cancer.data, dtype=torch.float)
    y = torch.tensor(breast_cancer.target)
    n_classes = len(torch.unique(y))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = Adaboost(num_classifiers=20)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)
