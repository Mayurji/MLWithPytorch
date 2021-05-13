import torch

class MeanSquareLoss:
    def __init__(self): pass

    def loss(self, y, y_pred):
        return torch.sum(torch.power((y - y_pred), 2),dim=1) / y.shape[0]

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class CrossEntropy:
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * torch.log(p) - (1 - y) * torch.log(1 - p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = torch.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

class MeanAbsoluteLoss:
    def __init__(self): pass

    def loss(self, y, y_pred):
        return torch.sum(torch.abs(y - y_pred), dim=1) / y.shape[0]

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class HuberLoss:
    def __init__(self):pass

    def loss(self, y, y_pred, delta):
        if torch.abs(y - y_pred) <=delta:
            return 0.5 * torch.pow(y - y_pred, 2)
        else:
            return (delta * torch.abs(y - y_pred)) - (0.5 * torch.pow(delta, 2))

class HingeLoss:
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        return torch.max(0, (1-y) * y_pred).values

class KLDivergence:
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        return torch.sum(y_pred * torch.log((y_pred / y)))
