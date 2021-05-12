import torch

class StochasticGradientDescentWithMomentum:
    def __init__(self, learning_rate=0.001, momentum=0):
        self.lr = learning_rate
        self.momentum = momentum
        self.w_update = None

    def update(self, w, gradient_wrt_w):
        if self.w_update is None:
            self.w_update = torch.zeros(w.shape)

        self.w_update = self.momentum * self.w_update + (1 - self.momentum) * gradient_wrt_w
        return w - self.lr * self.w_update

class NesterovAcceleratedGradient:
    def __init__(self, learning_rate=0.001, momentum=0.4):
        self.lr = learning_rate
        self.momentum = momentum
        self.w_update = torch.tensor([])

    def update(self, w, gradient_function):
        approx_future_gradient = torch.clip(gradient_function(w - self.momentum * self.w_update), -1, 1)

        if not self.w_update.any():
            self.w_update = torch.zeros(w.shape)

        self.w_update = self.momentum * self.w_update + self.lr * approx_future_gradient
        return w - self.w_update

class Adagrad:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.G = None
        self.eps = 1e-8

    def update(self, w, gradient_wrt_w):
        if self.G is None:
            self.G = torch.zeros(w.shape)

        self.G += torch.pow(gradient_wrt_w, 2)
        return w - self.lr * gradient_wrt_w / torch.sqrt(self.G + self.eps)

class Adadelta:
    def __init__(self, rho=0.95, eps=1e-6):
        self.E_W_update = None
        self.E_gradient = None
        self.w_update = None
        self.eps = eps
        self.rho = rho

    def update(self, w, gradient_wrt_w):
        if self.w_update is None:
            self.w_update = torch.zeros(w.shape)
            self.E_gradient = torch.zeros(gradient_wrt_w.shape)
            self.E_W_update = torch.zeros(w.shape)

        self.E_gradient = self.rho * self.E_gradient + (1 - self.rho) * torch.pow(gradient_wrt_w, 2)
        RMS_Delta_W = torch.sqrt(self.E_W_update + self.eps)
        RMS_gradient = torch.sqrt(self.E_gradient + self.eps)

        adaptive_lr = RMS_Delta_W / RMS_gradient
        self.w_update = adaptive_lr * gradient_wrt_w
        self.E_W_update = self.rho * self.E_W_update + (1 - self.rho) * torch.pow(self.w_update, 2)
        return w - self.w_update

class RMSprop:
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.lr = learning_rate
        self.Eg = None
        self.eps = 1e-8
        self.rho = rho

    def update(self, w, gradient_wrt_w):
        if self.Eg is None:
            self.Eg = torch.zeros(gradient_wrt_w.shape)

        self.Eg = self.rho * self.Eg + (1 -  self.rho) * torch.pow(gradient_wrt_w, 2)
        return w - self.lr * gradient_wrt_w / torch.sqrt(self.Eg + self.eps)

class Adam:
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.lr = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        self.b1 = b1
        self.b2 = b2

    def update(self, w, gradient_wrt_w):
        if self.m is None:
            self.m = torch.zeros(gradient_wrt_w.shape)
            self.v = torch.zeros(gradient_wrt_w.shape)

        self.m = self.b1 * self.m + (1 - self.b1) * gradient_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * torch.pow(gradient_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_update = self.lr * m_hat / torch.sqrt(v_hat) + self.eps

        return w - self.w_update
