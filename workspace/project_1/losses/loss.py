import numpy as np

class show_loss:
    def __call__(self, output, target):
        raise NotImplementedError

class MSEloss(show_loss):
    def __call__(self, output, target):
        return np.mean((output - target) ** 2) / 2

class BCELoss(show_loss):
    def __call__(self, output, target):
        eps = 1e-15
        output = np.clip(output, eps, 1 - eps)
        return -np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))