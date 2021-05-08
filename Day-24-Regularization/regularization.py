import torch
from sklearn.datasets import load_iris

class Regularization:
    def __init__(self, X):
        self.X = X

    def dropout(self, drop_probability):
        """
        Dropout is a regularization technique for neural networks that drops a unit (along with connections) at
        training time with a specified probability P (a common value is P = 0.5). At test time, all units are present,
        but with weights scaled by p(i.e. w becomes pw ).
        The idea is to prevent co-adaptation, where the neural network becomes too reliant on particular
        connections, as this could be symptomatic of overfitting. Intuitively, dropout can be thought of as creating
         an implicit ensemble of neural networks.
        :param drop_probability: float value between 0 to 1
        """
        if drop_probability < 1.0:
            keep_probability = 1 - drop_probability
        masker = torch.FloatTensor(self.X.shape).uniform_(0, 1)
        masked = masker < keep_probability

        if keep_probability > 0.0:
            scale = 1 / keep_probability
        else:
            scale = 0.0

        return masked * self.X * scale

    def L2_Regularization(self, y, W, lambda_value):
        """
        Weight Decay, or L2 Regularization, is a regularization technique applied to the weights of a neural network.
        We minimize a loss function compromising both the primary loss function and a penalty on the L2 Norm of the
        weights:
                L_new(w) = L_original(w) + lambda * W_T * W
        where  is a value determining the strength of the penalty (encouraging smaller weights).
        Weight decay can be incorporated directly into the weight update rule, rather than just implicitly by defining
        it through to objective function. Often weight decay refers to the implementation where we specify it directly
        in the weight update rule (whereas L2 regularization is usually the implementation which is specified in the
        objective function).
        """
        Regularization_term = (lambda_value * torch.mm(W, W.T)).type(torch.DoubleTensor) / (2 * y.shape[0])
        output = torch.sum((y - torch.mm(X, W.T))**2, dim=0) + Regularization_term
        return output

    def L1_Regularization(self, y, W, lambda_value):
        """
         L1 Regularization is a regularization technique applied to the weights of a neural network. We minimize a loss
        function compromising both the primary loss function and a penalty on the L1 Norm of the weights:
            L_new(w) = L_original(w) + lambda * ||W||
        where is a value determining the strength of the penalty. In contrast to weight decay, regularization promotes
        sparsity; i.e. some parameters have an optimal value of zero.
        """
        Regularization_term = torch.sum((lambda_value * torch.abs(W)).type(torch.DoubleTensor) / (2 * y.shape[0]),dim=1)
        output = torch.sum((y - torch.mm(X, W.T))**2, dim=0) + Regularization_term
        return output


if __name__ == '__main__':
    """
    Dropout: 
    
    A = torch.arange(20).reshape((5, 4))
    print(A)
    Regularizer = Regularization(X=A)
    print(Regularizer.dropout(drop_probability=0.5))
    
    L2 Regularization or Weight Decay:
    
    data = load_iris()
    X = torch.tensor(data.data, dtype=torch.double)
    y = torch.tensor(data.target).unsqueeze(1)
    W = torch.FloatTensor(X.shape[1]).uniform_(0, 1).unsqueeze(0).type(torch.DoubleTensor)
    Regularizer = Regularization(X)
    Regularizer.L2_Regularization(y=y, W=W, lambda_value=0.7)

    L1 Regularization:

    data = load_iris()
    X = torch.tensor(data.data, dtype=torch.double)
    y = torch.tensor(data.target).unsqueeze(1)
    W = torch.FloatTensor(X.shape[1]).uniform_(0, 1).unsqueeze(0).type(torch.DoubleTensor)
    Regularizer = Regularization(X)
    print(Regularizer.L1_Regularization(y=y, W=W, lambda_value=0.7))
    """
