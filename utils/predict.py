import numpy as np
import matplotlib.pyplot as plt

def save_model_params(params_dict, filepath="results/model_params.npz"):
    """使用 NumPy 的 .npz 文件格式保存模型参数"""
    np.savez(filepath, **params_dict)
    print(f"模型参数已成功保存至: {filepath}")

def load_model_params(filepath="results/model_params.npz"):
    """从 .npz 文件中加载模型参数"""
    loaded = np.load(filepath)
    params = {key: loaded[key] for key in loaded.files}
    print(f"模型参数已从 {filepath} 加载成功！")
    return params

def predict_and_show(model_forward_fn, X_test, y_test=None, num_samples=5):
    """
    模型预测与可视化展示函数（展示真实数字与预测数字）
    """
    # 随机抽取指定数量的测试样本
    indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
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

    # 处理图像用于画图展示
    if sample_images.ndim == 2 and sample_images.shape[1] == 784:
        vis_images = sample_images.reshape(-1, 28, 28)
    else:
        vis_images = sample_images.squeeze()

    # 使用 Matplotlib 可视化展示手写数字图片与预测
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(vis_images[i], cmap='gray')
        t_val = true_labels[i] if true_labels is not None else '?'
        plt.title(f"True: {t_val}\nPred: {pred_labels[i]}", fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()