"""成员 4 交付内容演示脚本。

直接运行（在项目根目录）::

    python part4_demo.py

演示内容：

1. ``MSELoss`` 的损失与梯度；
2. ``CrossEntropyLoss`` 的损失与梯度（logits / 概率 / 软标签）；
3. Accuracy、Precision、Recall、F1 与分类报告；
4. 用 ``CrossEntropyLoss`` 训练一个 softmax 回归模型，
   验证手写梯度确实能让损失下降、准确率上升。

整个脚本只依赖 NumPy 和本小组作业中的模块。
"""

import numpy as np

from core.losses import CrossEntropyLoss, MSELoss, build_loss
from utils.gradient_check import check_gradient
from utils.metrics import (
    calculate_accuracy,
    calculate_f1,
    calculate_metrics,
    calculate_precision,
    calculate_recall,
    classification_report,
    confusion_matrix,
)

SEP = "=" * 70


def _title(text):
    print(f"\n{SEP}\n{text}\n{SEP}")


# ----------------------------------------------------------------------
# 1. 均方误差
# ----------------------------------------------------------------------
def demo_mse():
    _title("1. 均方误差 MSELoss")

    pred = np.array([[0.2, -0.4, 0.6], [0.1, 0.3, -0.2]])
    target = np.array([[0.0, 0.0, 1.0], [0.0, 0.5, 0.0]])

    loss = MSELoss()
    value = loss.forward(pred, target)
    grad = loss.backward()

    print(f"pred   =\n{pred}")
    print(f"target =\n{target}")
    print(f"\nMSE loss = {value:.6f}")
    print(f"dL/dpred =\n{np.round(grad, 6)}")

    result = check_gradient(lambda p: loss.forward(p, target), pred, grad)
    print(
        f"\n数值梯度检查：最大相对误差 {result['max_rel_error']:.2e}"
        f" -> {'通过' if result['ok'] else '未通过'}"
    )


# ----------------------------------------------------------------------
# 2. 交叉熵
# ----------------------------------------------------------------------
def demo_cross_entropy():
    _title("2. 交叉熵 CrossEntropyLoss")

    logits = np.array(
        [[2.0, 0.5, -1.0], [0.1, 1.8, 0.2], [-0.5, 0.2, 2.4]]
    )
    labels = np.array([0, 1, 2])

    loss = build_loss("cross_entropy")
    value = loss.forward(logits, labels)
    grad = loss.backward()

    print(f"logits =\n{logits}")
    print(f"labels = {labels}")
    print(f"\nCE loss = {value:.6f}")
    print(f"dL/dlogits =\n{np.round(grad, 6)}")
    print(f"每行梯度之和 = {np.round(grad.sum(axis=1), 12)}")

    result = check_gradient(
        lambda z: loss.forward(z, labels), logits, grad
    )
    print(
        f"\n数值梯度检查：最大相对误差 {result['max_rel_error']:.2e}"
        f" -> {'通过' if result['ok'] else '未通过'}"
    )

    # 软标签
    soft_target = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]
    )
    soft_loss = CrossEntropyLoss()
    print(
        f"\n软标签 CE loss = "
        f"{soft_loss.forward(logits, soft_target):.6f}"
    )

    # 概率输入（例如网络末端已经做了 softmax）
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    prob_loss = CrossEntropyLoss(from_logits=False)
    print(
        f"概率输入 CE loss = "
        f"{prob_loss.forward(probabilities, labels):.6f}"
        f"（应与上面的 logits 版本一致）"
    )


# ----------------------------------------------------------------------
# 3. 评价指标
# ----------------------------------------------------------------------
def demo_metrics():
    _title("3. 评价指标：Accuracy / Precision / Recall / F1")

    y_true = np.array([0, 1, 2, 2, 1, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 2, 2, 1, 1, 0, 1, 0, 0, 2])

    print("混淆矩阵（行 = 真实类别，列 = 预测类别）：")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nAccuracy = {calculate_accuracy(y_true, y_pred):.6f}")
    print(
        "宏平均 Precision / Recall / F1 = "
        f"{calculate_precision(y_true, y_pred):.6f} / "
        f"{calculate_recall(y_true, y_pred):.6f} / "
        f"{calculate_f1(y_true, y_pred):.6f}"
    )
    print(
        f"微平均 F1 = {calculate_f1(y_true, y_pred, average='micro'):.6f}"
        "（单标签多分类中等于 Accuracy）"
    )
    print("\n分类报告：")
    print(classification_report(y_true, y_pred, digits=4))

    # 二分类正类指标
    binary_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    binary_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    print("\n二分类（average='binary'，正类为 1）：")
    print(f"  Accuracy  = {calculate_accuracy(binary_true, binary_pred):.6f}")
    print(
        f"  Precision = "
        f"{calculate_precision(binary_true, binary_pred, average='binary'):.6f}"
    )
    print(
        f"  Recall    = "
        f"{calculate_recall(binary_true, binary_pred, average='binary'):.6f}"
    )
    print(
        f"  F1        = "
        f"{calculate_f1(binary_true, binary_pred, average='binary'):.6f}"
    )

    # 可以直接传入 (N, C) 的分数矩阵，内部自动 argmax
    scores = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
    )
    scores_true = np.array([0, 1, 2])
    print(
        "\n分数矩阵输入的 Accuracy = "
        f"{calculate_accuracy(scores_true, scores):.6f}"
    )


# ----------------------------------------------------------------------
# 4. 用交叉熵梯度训练一个小模型
# ----------------------------------------------------------------------
def _make_blobs(n_per_class=80, n_classes=3, seed=42):
    """生成三个线性可分的二维高斯簇。"""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n_classes, endpoint=False)
    centers = 3.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)

    features = []
    labels = []
    for index, center in enumerate(centers):
        features.append(rng.normal(center, 0.7, size=(n_per_class, 2)))
        labels.append(np.full(n_per_class, index, dtype=np.int64))

    x = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    order = rng.permutation(y.size)
    return x[order], y[order]


def demo_training():
    _title("4. 用 CrossEntropyLoss 的梯度训练 softmax 回归")

    x, y = _make_blobs()
    n_features = x.shape[1]
    n_classes = int(y.max()) + 1

    # 从 0 初始化：初始 logits 全为 0，三个类别概率相同
    weights = np.zeros((n_features, n_classes))
    bias = np.zeros((1, n_classes))

    loss_fn = CrossEntropyLoss()
    learning_rate = 0.5
    first_loss = None

    for epoch in range(101):
        logits = x @ weights + bias
        loss_value = loss_fn.forward(logits, y)
        grad = loss_fn.backward()

        # dL/dW = X^T @ dL/dlogits，dL/db = sum(dL/dlogits)
        weights -= learning_rate * (x.T @ grad)
        bias -= learning_rate * np.sum(grad, axis=0, keepdims=True)

        if epoch == 0:
            first_loss = loss_value
        if epoch % 20 == 0:
            accuracy = calculate_accuracy(y, logits)
            print(
                f"epoch {epoch:3d}   loss={loss_value:.4f}   "
                f"accuracy={accuracy:.4f}"
            )

    logits = x @ weights + bias
    metrics = calculate_metrics(y, logits)
    print(
        f"\n训练完成：loss {first_loss:.4f} -> {loss_fn.last_loss:.4f}"
    )
    print(
        f"最终指标：Accuracy={metrics['accuracy']:.4f}  "
        f"Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}"
    )
    assert metrics["accuracy"] > 0.9, "损失梯度可能有问题：准确率未达到 0.9"
    print("梯度方向正确：损失下降，准确率超过 0.9。")


def main():
    print(SEP)
    print("成员 4：损失函数与评价指标 —— 交付演示")
    print(SEP)
    demo_mse()
    demo_cross_entropy()
    demo_metrics()
    demo_training()
    print(f"\n{SEP}\n全部演示结束。\n{SEP}")


if __name__ == "__main__":
    main()
