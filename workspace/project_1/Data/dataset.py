import numpy as np


class show_DataSet:
    def __init__(self, X=None, y=None, batch_size=32, shuffle=True, filepath=None):
        if filepath is not None:
            X, y = self._load_csv(filepath)

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.indices = np.arange(self.n_samples)
        self.current_idx = 0

    def _load_csv(self, filepath):
        print(f"正在读取CSV文件: {filepath}")
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        print(f"成功读取数据，形状: {data.shape}")

        X = data[:, [0, 1]]
        y = data[:, 2].reshape(-1, 1)

        print(f"加载了 {X.shape[0]} 个样本，{X.shape[1]} 个特征")
        print(f"标签分布: 0类={np.sum(y == 0)}, 1类={np.sum(y == 1)}")

        # 标准化
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_scaled = (X - X_mean) / X_std

        print(f"特征均值: {X_mean}")
        print(f"特征标准差: {X_std}")
        return X_scaled, y

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration
        end_idx = min(self.current_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[self.current_idx:end_idx]
        self.current_idx = end_idx
        return self.X[batch_indices], self.y[batch_indices]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size