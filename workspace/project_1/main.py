import numpy as np
import matplotlib.pyplot as plt

from Data import show_DataSet
from modeling import show_Model
from losses import MSEloss, BCELoss
from optimizer import show_SGD, show_Momentum, show_NAG, show_Adagrad, show_RMSprop, show_AdaDelta, show_Adam
from utils import train_test_split_numpy
from metrics import accuracy_score_numpy
from postprocess import evaluate_model

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
                    f'Epoch {epoch + 1}/{epochs_logistic}, '
                    f'Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {acc:.4f}')

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
            f"{opt_name:<15} {data['final_accuracy']:.4f}             "
            f" {data['final_train_loss']:.4f}            "
            f"  {data['final_test_loss']:.4f}{mark}")

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

        # 2. 线性回归 - 直方图（年龄分布，按是否购买分组）
        ax2 = plt.subplot(2, 3, 2)

        # 分离两类数据
        age_0 = X_test[y_test.flatten() == 0][:, 0]
        age_1 = X_test[y_test.flatten() == 1][:, 0]

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
                if wrong_y[i] == 0:
                    ax5.scatter(wrong_X[i, 0], wrong_X[i, 1],
                                c='blue', marker='x', s=80, alpha=0.9,
                                linewidths=2,
                                label='错误: 实际0→预测1' if i == 0 else "")
                else:
                    ax5.scatter(wrong_X[i, 0], wrong_X[i, 1],
                                c='red', marker='x', s=80, alpha=0.9,
                                linewidths=2,
                                label='错误: 实际1→预测0' if i == 0 else "")

        ax5.set_title(f'逻辑回归 - 分类结果散点图 (最佳: {best_opt[0]})\n○=正确分类  ×=错误分类')
        ax5.set_xlabel('年龄 (标准化)')
        ax5.set_ylabel('薪资 (标准化)')
        ax5.legend(loc='upper left', fontsize=7)
        ax5.grid(True, alpha=0.3)

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