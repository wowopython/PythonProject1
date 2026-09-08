import numpy as np

def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    test_size_int = int(n_samples * test_size)
    test_indices = indices[:test_size_int]
    train_indices = indices[test_size_int:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]