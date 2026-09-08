import numpy as np

class show_Optimizer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def zero_grad(self):
        self.model.zero_grad()

    def step(self):
        self._update_params()

    def _update_params(self):
        raise NotImplementedError

class show_SGD(show_Optimizer):
    def __init__(self, model, lr=0.01):
        super().__init__(model, lr)

    def _update_params(self):
        if self.model.grad_W is not None:
            self.model.W -= self.lr * self.model.grad_W
            self.model.b -= self.lr * self.model.grad_b

class show_Momentum(show_Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.9):
        super().__init__(model, lr)
        self.momentum = momentum
        self.v_W = np.zeros_like(model.W)
        self.v_b = np.zeros_like(model.b)

    def _update_params(self):
        if self.model.grad_W is not None:
            self.v_W = self.momentum * self.v_W - self.lr * self.model.grad_W
            self.v_b = self.momentum * self.v_b - self.lr * self.model.grad_b
            self.model.W += self.v_W
            self.model.b += self.v_b

class show_NAG(show_Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.9):
        super().__init__(model, lr)
        self.momentum = momentum
        self.v_W = np.zeros_like(model.W)
        self.v_b = np.zeros_like(model.b)

    def _update_params(self):
        if self.model.grad_W is not None:
            self.v_W = self.momentum * self.v_W - self.lr * self.model.grad_W
            self.v_b = self.momentum * self.v_b - self.lr * self.model.grad_b
            self.model.W += self.v_W
            self.model.b += self.v_b

class show_Adagrad(show_Optimizer):
    def __init__(self, model, lr=0.01, eps=1e-8):
        super().__init__(model, lr)
        self.eps = eps
        self.G_W = np.zeros_like(model.W)
        self.G_b = np.zeros_like(model.b)

    def _update_params(self):
        if self.model.grad_W is not None:
            self.G_W += self.model.grad_W ** 2
            self.G_b += self.model.grad_b ** 2
            self.model.W -= self.lr / (np.sqrt(self.G_W) + self.eps) * self.model.grad_W
            self.model.b -= self.lr / (np.sqrt(self.G_b) + self.eps) * self.model.grad_b

class show_RMSprop(show_Optimizer):
    def __init__(self, model, lr=0.01, decay=0.9, eps=1e-8):
        super().__init__(model, lr)
        self.decay = decay
        self.eps = eps
        self.E_W = np.zeros_like(model.W)
        self.E_b = np.zeros_like(model.b)

    def _update_params(self):
        if self.model.grad_W is not None:
            self.E_W = self.decay * self.E_W + (1 - self.decay) * self.model.grad_W ** 2
            self.E_b = self.decay * self.E_b + (1 - self.decay) * self.model.grad_b ** 2
            self.model.W -= self.lr / (np.sqrt(self.E_W) + self.eps) * self.model.grad_W
            self.model.b -= self.lr / (np.sqrt(self.E_b) + self.eps) * self.model.grad_b

class show_AdaDelta(show_Optimizer):
    def __init__(self, model, rho=0.95, eps=1e-6):
        super().__init__(model, lr=1.0)
        self.rho = rho
        self.eps = eps
        self.E_W = np.zeros_like(model.W)
        self.E_b = np.zeros_like(model.b)
        self.delta_W = np.zeros_like(model.W)
        self.delta_b = np.zeros_like(model.b)

    def _update_params(self):
        if self.model.grad_W is not None:
            self.E_W = self.rho * self.E_W + (1 - self.rho) * self.model.grad_W ** 2
            self.E_b = self.rho * self.E_b + (1 - self.rho) * self.model.grad_b ** 2

            update_W = -np.sqrt(self.delta_W + self.eps) / (np.sqrt(self.E_W + self.eps)) * self.model.grad_W
            update_b = -np.sqrt(self.delta_b + self.eps) / (np.sqrt(self.E_b + self.eps)) * self.model.grad_b

            self.model.W += update_W
            self.model.b += update_b

            self.delta_W = self.rho * self.delta_W + (1 - self.rho) * update_W ** 2
            self.delta_b = self.rho * self.delta_b + (1 - self.rho) * update_b ** 2

class show_Adam(show_Optimizer):
    def __init__(self, model, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(model, lr)
        self.betas = betas
        self.eps = eps
        self.m_W = np.zeros_like(model.W)
        self.m_b = np.zeros_like(model.b)
        self.v_W = np.zeros_like(model.W)
        self.v_b = np.zeros_like(model.b)
        self.t = 0

    def _update_params(self):
        if self.model.grad_W is not None:
            self.t += 1
            self.m_W = self.betas[0] * self.m_W + (1 - self.betas[0]) * self.model.grad_W
            self.m_b = self.betas[0] * self.m_b + (1 - self.betas[0]) * self.model.grad_b
            self.v_W = self.betas[1] * self.v_W + (1 - self.betas[1]) * self.model.grad_W ** 2
            self.v_b = self.betas[1] * self.v_b + (1 - self.betas[1]) * self.model.grad_b ** 2

            m_W_hat = self.m_W / (1 - self.betas[0] ** self.t)
            m_b_hat = self.m_b / (1 - self.betas[0] ** self.t)
            v_W_hat = self.v_W / (1 - self.betas[1] ** self.t)
            v_b_hat = self.v_b / (1 - self.betas[1] ** self.t)

            self.model.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            self.model.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)