from pathlib import Path

import numpy as np

def save_model_params(params_dict, filepath="results/model_params.npz"):
    """使用 NumPy 的 .npz 文件格式保存模型参数"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.savez(filepath, **params_dict)
    print(f"模型参数已成功保存至: {filepath}")

def load_model_params(filepath="results/model_params.npz"):
    """从 .npz 文件中加载模型参数"""
    with np.load(filepath, allow_pickle=False) as loaded:
        params = {key: loaded[key] for key in loaded.files}
    print(f"模型参数已从 {filepath} 加载成功！")
    return params

def predict_and_show(model_forward_fn, X_test, y_test=None, num_samples=5,
                     output_path=None, show=True, seed=None):
    """
    模型预测与可视化展示函数（展示真实数字与预测数字）
    """
    import matplotlib.pyplot as plt

    if not 1 <= num_samples <= X_test.shape[0]:
        raise ValueError("num_samples 必须介于 1 和测试样本数量之间")
    indices = np.random.default_rng(seed).choice(X_test.shape[0], num_samples, replace=False)
    sample_images = X_test[indices]
    true_labels = y_test[indices] if y_test is not None else None

    # 执行模型前向推理预测
    outputs = model_forward_fn(sample_images)
    pred_labels = np.argmax(outputs, axis=1)

    # 打印控制台文字结果
    print("\n========== 手写数字识别预测结果 ==========")
    for i in range(num_samples):
        t_str = f"真实数字: {true_labels[i]}" if true_labels is not None else ""
        print(f"样本 [{i+1}] -> {t_str} | 预测数字: {pred_labels[i]}")
    print("==========================================\n")

    # 保留批次维；单张 CNN 图片不能使用 squeeze() 删除所有单例维度。
    vis_images = sample_images.reshape(num_samples, 28, 28)

    # 使用 Matplotlib 可视化展示手写数字图片与预测
    figure, axes = plt.subplots(1, num_samples, figsize=(2 * num_samples, 3), squeeze=False)
    for i in range(num_samples):
        axis = axes[0, i]
        axis.imshow(vis_images[i], cmap='gray')
        t_val = true_labels[i] if true_labels is not None else '?'
        axis.set_title(f"True: {t_val}\nPred: {pred_labels[i]}", fontsize=10)
        axis.axis('off')
    
    figure.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return figure
