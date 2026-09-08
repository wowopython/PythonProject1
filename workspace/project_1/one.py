import numpy as np
import matplotlib.pyplot as plt


# ==================== 数据加载模块 ====================
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


# ==================== 模型类 ====================
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


# ==================== 损失函数类 ====================
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


# ==================== 优化器基类 ====================
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


# ==================== 7种优化器 ====================
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


# ==================== 工具函数 ====================
def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    test_size_int = int(n_samples * test_size)
    test_indices = indices[:test_size_int]
    train_indices = indices[test_size_int:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def accuracy_score_numpy(y_true, y_pred):
    return np.mean(y_true.flatten() == y_pred.flatten())


def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    test_pred = model(X_test)
    test_pred_binary = (test_pred > 0.5).astype(int)

    # 计算准确率
    acc = np.mean(y_test.flatten() == test_pred_binary.flatten())

    # 计算混淆矩阵
    tp = np.sum((y_test == 1) & (test_pred_binary == 1))
    tn = np.sum((y_test == 0) & (test_pred_binary == 0))
    fp = np.sum((y_test == 0) & (test_pred_binary == 1))
    fn = np.sum((y_test == 1) & (test_pred_binary == 0))

    # 计算精确率、召回率、F1分数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return {
        'accuracy': acc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ==================== 主程序 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("PyTorch风格算法实现 - 线性回归与逻辑回归")
    print("=" * 60)

    # ==================== 1. 加载数据 ====================
    csv_file = r'D:\PythonProject1\workspace\Social_Network_Ads.csv'
    dataset = show_DataSet(filepath=csv_file, batch_size=32, shuffle=True)

    X_all = dataset.X
    y_all = dataset.y

    X_train, X_test, y_train, y_test = train_test_split_numpy(X_all, y_all, test_size=0.2, random_state=42)
    print(f"\n训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

    train_dataset = show_DataSet(X_train, y_train, batch_size=32, shuffle=True)

    # ==================== 2. 线性回归 ====================
    print("\n" + "=" * 60)
    print("1. 线性回归 (使用SGD优化器)")
    print("=" * 60)

    model_lr = show_Model(input_dim=2, output_dim=1, model_type='linear')
    optimizer_lr = show_SGD(model_lr, lr=0.01)
    loss_fn_lr = MSEloss()

    epochs_lr = 100
    train_losses_lr = []
    test_losses_lr = []

    for epoch in range(epochs_lr):
        epoch_loss = 0
        for batch_X, batch_y in train_dataset:
            optimizer_lr.zero_grad()
            output = model_lr.forward_with_cache(batch_X)
            loss = loss_fn_lr(output, batch_y)
            model_lr.backward(output, batch_y)
            optimizer_lr.step()
            epoch_loss += loss

        avg_train_loss = epoch_loss / len(train_dataset)
        train_losses_lr.append(avg_train_loss)

        # 计算测试损失
        test_output = model_lr(X_test)
        test_loss = loss_fn_lr(test_output, y_test)
        test_losses_lr.append(test_loss)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1}/{epochs_lr}, Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}')

    print(f"\n线性回归训练完成！最终训练损失: {train_losses_lr[-1]:.6f}")
    print(f"最终测试损失: {test_losses_lr[-1]:.6f}")
    print(f"权重: {model_lr.W.flatten()}")
    print(f"偏置: {model_lr.b.flatten()}")

    # ==================== 3. 逻辑回归 - 测试所有优化器 ====================
    print("\n" + "=" * 60)
    print("2. 逻辑回归 - 7种优化器对比")
    print("=" * 60)

    optimizers_config = {
        'SGD': (show_SGD, {'lr': 0.1}),
        'Momentum': (show_Momentum, {'lr': 0.1, 'momentum': 0.9}),
        'NAG': (show_NAG, {'lr': 0.1, 'momentum': 0.9}),
        'Adagrad': (show_Adagrad, {'lr': 0.1}),
        'RMSprop': (show_RMSprop, {'lr': 0.01, 'decay': 0.9}),
        'AdaDelta': (show_AdaDelta, {'rho': 0.95}),
        'Adam': (show_Adam, {'lr': 0.01})
    }

    results = {}
    epochs_logistic = 100

    for opt_name, (opt_class, opt_params) in optimizers_config.items():
        print(f"\n训练优化器: {opt_name}")
        print("-" * 40)

        model = show_Model(input_dim=2, output_dim=1, model_type='logistic')
        optimizer = opt_class(model, **opt_params)
        loss_fn = BCELoss()

        train_losses = []
        test_losses = []
        accuracies = []

        for epoch in range(epochs_logistic):
            epoch_loss = 0
            for batch_X, batch_y in train_dataset:
                optimizer.zero_grad()
                output = model.forward_with_cache(batch_X)
                loss = loss_fn(output, batch_y)
                model.backward(output, batch_y)
                optimizer.step()
                epoch_loss += loss

            avg_train_loss = epoch_loss / len(train_dataset)
            train_losses.append(avg_train_loss)

            test_output = model(X_test)
            test_loss = loss_fn(test_output, y_test)
            test_losses.append(test_loss)

            predictions = (test_output > 0.5).astype(int)
            acc = accuracy_score_numpy(y_test, predictions)
            accuracies.append(acc)

            if (epoch + 1) % 20 == 0:
                print(
                    f'Epoch {epoch + 1}/{epochs_logistic}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {acc:.4f}')

        results[opt_name] = {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1] if accuracies else 0,
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_test_loss': test_losses[-1] if test_losses else 0
        }

        print(f"\n{opt_name} 完成！最终准确率: {results[opt_name]['final_accuracy']:.4f}")

    # ==================== 4. 结果汇总 ====================
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    best_opt = max(results.items(), key=lambda x: x[1]['final_accuracy'])

    print(f"{'优化器':<15} {'最终准确率':<20} {'训练损失':<20} {'测试损失':<20}")
    print("-" * 75)
    for opt_name, data in results.items():
        mark = " ★ 最佳" if opt_name == best_opt[0] else ""
        print(
            f"{opt_name:<15} {data['final_accuracy']:.4f}              {data['final_train_loss']:.4f}              {data['final_test_loss']:.4f}{mark}")

    print(f"\n最佳优化器: {best_opt[0]} 准确率: {best_opt[1]['final_accuracy']:.4f}")

    # ==================== 5. 可视化 ====================
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建2x3的子图布局
        fig = plt.figure(figsize=(18, 12))

        # ===== 第1行：线性回归结果 =====
        # 1. 线性回归 - 损失曲线
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(train_losses_lr, label='训练损失', linewidth=2, color='blue')
        ax1.plot(test_losses_lr, label='测试损失', linewidth=2, color='red')
        ax1.set_title('线性回归 - 损失曲线 (SGD)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 线性回归 - 直方图（年龄分布，按薪资分组）
        ax2 = plt.subplot(2, 3, 2)

        # 分离两类数据
        age_0 = X_test[y_test.flatten() == 0][:, 0]  # 未购买的年龄
        age_1 = X_test[y_test.flatten() == 1][:, 0]  # 购买的年龄

        # 绘制直方图
        bins = np.linspace(min(X_test[:, 0]), max(X_test[:, 0]), 20)

        if len(age_0) > 0:
            ax2.hist(age_0, bins=bins, alpha=0.6, label='未购买 (实际=0)',
                     color='blue', edgecolor='black', linewidth=0.5)
        if len(age_1) > 0:
            ax2.hist(age_1, bins=bins, alpha=0.6, label='购买 (实际=1)',
                     color='red', edgecolor='black', linewidth=0.5)

        ax2.set_title('线性回归 - 年龄分布直方图 (按是否购买分组)')
        ax2.set_xlabel('年龄 (标准化)')
        ax2.set_ylabel('频数')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ===== 第2行：逻辑回归结果 =====
        # 3. 逻辑回归 - 训练损失对比
        ax3 = plt.subplot(2, 3, 3)
        for opt_name, data in results.items():
            ax3.plot(data['train_losses'], label=opt_name, linewidth=1.5)
        ax3.set_title('逻辑回归 - 训练损失对比')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('训练损失')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. 逻辑回归 - 测试损失对比
        ax4 = plt.subplot(2, 3, 4)
        for opt_name, data in results.items():
            ax4.plot(data['test_losses'], label=opt_name, linewidth=1.5)
        ax4.set_title('逻辑回归 - 测试损失对比')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('测试损失')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 5. 逻辑回归 - 散点图（年龄 vs 薪资，显示分类结果）
        ax5 = plt.subplot(2, 3, 5)

        # 获取最佳模型的预测结果
        best_model = best_opt[1]['model']
        test_pred_binary = (best_model(X_test) > 0.5).astype(int).flatten()

        # 分离正确分类和错误分类的样本
        correct_mask = (test_pred_binary == y_test.flatten())
        wrong_mask = ~correct_mask

        # 正确分类的样本
        correct_X = X_test[correct_mask]
        correct_y = y_test[correct_mask].flatten()

        # 错误分类的样本
        wrong_X = X_test[wrong_mask]
        wrong_y = y_test[wrong_mask].flatten()

        # 绘制正确分类的样本
        if len(correct_X) > 0:
            for i in range(len(correct_X)):
                if correct_y[i] == 0:
                    ax5.scatter(correct_X[i, 0], correct_X[i, 1],
                                c='blue', marker='o', s=50, alpha=0.7,
                                edgecolors='black', linewidth=0.5,
                                label='正确: 实际0→预测0' if i == 0 else "")
                else:
                    ax5.scatter(correct_X[i, 0], correct_X[i, 1],
                                c='red', marker='o', s=50, alpha=0.7,
                                edgecolors='black', linewidth=0.5,
                                label='正确: 实际1→预测1' if i == 0 else "")

        # 绘制错误分类的样本
        if len(wrong_X) > 0:
            for i in range(len(wrong_X)):
                if wrong_y[i] == 0:  # 实际0但预测1
                    ax5.scatter(wrong_X[i, 0], wrong_X[i, 1],
                                c='blue', marker='x', s=80, alpha=0.9,
                                linewidths=2,
                                label='错误: 实际0→预测1' if i == 0 else "")
                else:  # 实际1但预测0
                    ax5.scatter(wrong_X[i, 0], wrong_X[i, 1],
                                c='red', marker='x', s=80, alpha=0.9,
                                linewidths=2,
                                label='错误: 实际1→预测0' if i == 0 else "")

        ax5.set_title(f'逻辑回归 - 分类结果散点图 (最佳: {best_opt[0]})\n○=正确分类  ×=错误分类')
        ax5.set_xlabel('年龄 (标准化)')
        ax5.set_ylabel('薪资 (标准化)')
        ax5.legend(loc='upper left', fontsize=7)
        ax5.grid(True, alpha=0.3)

        # 添加图例说明
        ax5.text(0.02, 0.98, '蓝色=实际0(未购买)  红色=实际1(购买)',
                 transform=ax5.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 6. 逻辑回归 - 最终准确率柱状图
        ax6 = plt.subplot(2, 3, 6)
        opt_names = list(results.keys())
        final_accs = [results[opt]['final_accuracy'] for opt in opt_names]
        colors_bar = ['green' if i == np.argmax(final_accs) else 'skyblue' for i in range(len(opt_names))]
        bars = ax6.bar(opt_names, final_accs, color=colors_bar, edgecolor='black')
        ax6.set_title('逻辑回归 - 最终准确率对比')
        ax6.set_ylabel('准确率')
        ax6.set_ylim([0.5, 1.0])
        ax6.grid(True, alpha=0.3, axis='y')

        for bar, acc in zip(bars, final_accs):
            ax6.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                     f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('all_results_mixed.png', dpi=300, bbox_inches='tight')
        print("\n可视化图表已保存为 'all_results_mixed.png'")
        plt.show()
    except Exception as e:
        print(f"\n可视化提示: {e}")

    # ==================== 6. 最佳模型详细信息 ====================
    print("\n" + "=" * 60)
    print("最佳模型详细信息")
    print("=" * 60)

    best_model = best_opt[1]['model']

    # 使用evaluation模块评估
    eval_results = evaluate_model(best_model, X_test, y_test)

    print(f"最佳优化器: {best_opt[0]}")
    print(f"测试准确率: {eval_results['accuracy']:.4f}")
    print(f"权重: {best_model.W.flatten()}")
    print(f"偏置: {best_model.b.flatten()}")
    print(f"\n混淆矩阵:")
    print(f"  TP: {eval_results['tp']}, TN: {eval_results['tn']}")
    print(f"  FP: {eval_results['fp']}, FN: {eval_results['fn']}")
    print(f"  精确率: {eval_results['precision']:.4f}")
    print(f"  召回率: {eval_results['recall']:.4f}")
    print(f"  F1分数: {eval_results['f1']:.4f}")

    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)