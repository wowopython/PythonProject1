import numpy as np

class show_Model:
    def __init__(self, input_dim=2, output_dim=1, model_type='logistic'):
        self.model_type = model_type
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        self.grad_W = None
        self.grad_b = None
        self.X_cache = None

    def __call__(self, X):
        """前向传播"""
        z = np.dot(X, self.W) + self.b
        if self.model_type == 'linear':
            return z
        elif self.model_type == 'logistic':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward_with_cache(self, X):
        """前向传播并缓存输入"""
        self.X_cache = X
        return self.__call__(X)

    def zero_grad(self):
        self.grad_W = None
        self.grad_b = None

    def backward(self, output, target):
        n_samples = output.shape[0]
        grad = output - target
        if self.X_cache is not None:
            self.grad_W = np.dot(self.X_cache.T, grad) / n_samples
            self.grad_b = np.mean(grad, axis=0, keepdims=True)