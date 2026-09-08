import numpy as np

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