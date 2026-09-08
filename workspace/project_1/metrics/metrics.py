import numpy as np

def accuracy_score_numpy(y_true, y_pred):
    return np.mean(y_true.flatten() == y_pred.flatten())