"""评价指标（成员 4：损失函数与评价指标）。

对应最终项目目录中的 ``utils/metrics.py``。

提供的指标
----------
* 准确率：:func:`calculate_accuracy`
* 精确率：:func:`calculate_precision`
* 召回率：:func:`calculate_recall`
* F1 分数：:func:`calculate_f1`
* 混淆矩阵：:func:`confusion_matrix`
* 汇总报告：:func:`calculate_metrics`、:func:`classification_report`

输入约定
--------
* ``y_true``：形如 ``(N,)`` 的类别下标；也接受 one-hot / 概率矩阵 ``(N, C)``，
  此时按行取 ``argmax``；还接受 ``(N, 1)`` 的整数标签列。
* ``y_pred``：形如 ``(N,)`` 的预测类别下标；也接受模型输出 ``(N, C)``
  （logits 或概率均可），此时按行取 ``argmax``；``(N, 1)`` 的整数标签列同样支持。

多分类默认使用 ``average="macro"``（各类别指标取算术平均）。
需要二分类正类（标签 1）的指标时，显式传入 ``average="binary"``。

示例
----
>>> import numpy as np
>>> y_true = np.array([0, 1, 2, 2, 1, 0, 1, 2, 0, 1])
>>> y_pred = np.array([0, 2, 2, 1, 1, 0, 1, 0, 0, 2])
>>> round(calculate_accuracy(y_true, y_pred), 4)
0.6
>>> round(calculate_f1(y_true, y_pred), 4)
0.5873
"""

import numpy as np

__all__ = [
    "labels_from_scores",
    "confusion_matrix",
    "calculate_accuracy",
    "calculate_precision",
    "calculate_recall",
    "calculate_f1",
    "calculate_metrics",
    "classification_report",
]

_AVERAGES = ("binary", "macro", "micro", "weighted")


# ----------------------------------------------------------------------
# 输入处理
# ----------------------------------------------------------------------
def labels_from_scores(y):
    """把 ``(N, C)`` 的分数矩阵按行 ``argmax`` 转成 ``(N,)`` 的类别下标。

    一维输入视为已经是类别下标，原样返回（转成 ``np.ndarray``）。
    """
    arr = np.asarray(y)
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError("输入不能为空")
        return np.argmax(arr, axis=1)
    if arr.ndim == 1:
        return arr
    raise ValueError(f"只支持一维类别下标或二维分数矩阵，收到形状 {arr.shape}")


def _validate_int_values(values, name):
    """校验一维浮点数组可以当作整数类别标签。"""
    if values.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} 中不能包含 inf / nan")
    if np.any(values != np.floor(values)):
        raise ValueError(f"{name} 中的类别标签必须是整数")
    return values.astype(np.int64)


def _as_int_labels(y, name):
    """校验并转换类别标签，返回 int64 的一维数组。

    除了 ``(N,)`` 和 ``(N, C)``（one-hot / 分数矩阵），也接受
    ``(N, 1)`` 的整数标签列（例如 ``labels.reshape(-1, 1)`` 之后的结果）。
    """
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{name} 中的类别标签必须是数值")
        values = arr.astype(np.float64, copy=False).reshape(-1)
        if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError(
                f"{name} 的形状为 (N, 1) 但不是整数标签列；"
                "二分类概率请先转换为 0/1 标签，或传入 (N, 2) 的概率矩阵"
            )
        return _validate_int_values(values, name)

    arr = labels_from_scores(arr).reshape(-1)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} 中的类别标签必须是数值")
    return _validate_int_values(arr.astype(np.float64, copy=False), name)


def _prepare(y_true, y_pred):
    """校验 ``y_true`` / ``y_pred`` 并返回整数标签数组。"""
    true = _as_int_labels(y_true, "y_true")
    pred = _as_int_labels(y_pred, "y_pred")
    if true.size != pred.size:
        raise ValueError(
            f"y_true 与 y_pred 的样本数必须相同，收到 {true.size} 和 {pred.size}"
        )
    return true, pred


def _resolve_labels(labels, true, pred):
    """确定参与统计的类别列表。``labels=None`` 时取出现过的类别（升序）。"""
    observed = np.unique(np.concatenate((true, pred)))
    if labels is None:
        return observed

    arr = np.asarray(labels)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("labels 必须是长度为 C 的一维数组")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("labels 中的类别标签必须是数值")
    values = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
        raise ValueError("labels 中的类别标签必须是整数")
    values = values.astype(np.int64)
    if np.unique(values).size != values.size:
        raise ValueError("labels 中不能有重复的类别")

    missing = np.setdiff1d(observed, values)
    if missing.size > 0:
        raise ValueError(f"标签 {missing.tolist()} 未包含在 labels 中")
    return values


def _validate_average(average):
    if average is None or average in _AVERAGES:
        return average
    raise ValueError(
        f"average 必须是 None、'binary'、'macro'、'micro' 或 'weighted'，"
        f"收到 {average!r}"
    )


def _safe_div(numerator, denominator, zero_division=0.0):
    """逐元素除法，分母为 0 时取 ``zero_division``。"""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.zeros(np.broadcast(num, den).shape, dtype=np.float64)
    np.divide(num, den, out=out, where=den != 0)
    if np.any(den == 0):
        out = np.where(den == 0, float(zero_division), out)
    return out


# ----------------------------------------------------------------------
# 混淆矩阵与逐类统计
# ----------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, labels=None):
    """计算混淆矩阵，**行是真实类别，列是预测类别**。

    参数
    ----
    y_true, y_pred : array_like
        真实 / 预测标签，规则见模块文档。
    labels : array_like, optional
        统计的类别及顺序；默认使用出现过的类别（升序）。
        需要固定 0~9 十个类别时可传入 ``labels=range(10)``。

    返回
    ----
    np.ndarray
        形状 ``(C, C)``、dtype 为 ``int64`` 的混淆矩阵，
        ``cm[i, j]`` 表示真实类别 ``labels[i]`` 被预测为 ``labels[j]`` 的次数。

    示例
    ----
    >>> confusion_matrix([0, 1, 1], [0, 1, 0])
    array([[1, 0],
           [1, 1]])
    """
    cm, _ = _confusion(y_true, y_pred, labels)
    return cm


def _confusion(y_true, y_pred, labels=None):
    """计算混淆矩阵，同时返回实际使用的类别顺序。"""
    true, pred = _prepare(y_true, y_pred)
    labels_arr = _resolve_labels(labels, true, pred)

    index = {int(label): i for i, label in enumerate(labels_arr)}
    true_idx = np.fromiter(
        (index[int(v)] for v in true), dtype=np.int64, count=true.size
    )
    pred_idx = np.fromiter(
        (index[int(v)] for v in pred), dtype=np.int64, count=pred.size
    )

    cm = np.zeros((labels_arr.size, labels_arr.size), dtype=np.int64)
    np.add.at(cm, (true_idx, pred_idx), 1)
    return cm, labels_arr


def _per_class(y_true, y_pred, labels=None, zero_division=0):
    """返回逐类的 precision / recall / f1 / support 等信息。"""
    cm, labels_arr = _confusion(y_true, y_pred, labels)

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)   # 真实样本数
    predicted = cm.sum(axis=0).astype(np.float64)  # 预测样本数
    fp = predicted - tp
    fn = support - tp

    precision = _safe_div(tp, tp + fp, zero_division)
    recall = _safe_div(tp, tp + fn, zero_division)
    f1 = _safe_div(2.0 * precision * recall, precision + recall, zero_division)

    return {
        "labels": labels_arr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _aggregate(metric, stats, average, zero_division=0.0):
    """按 ``average`` 汇总逐类指标。"""
    average = _validate_average(average)
    values = stats[metric]

    if average is None:
        return values
    if average == "macro":
        return float(np.mean(values))
    if average == "weighted":
        total = stats["support"].sum()
        return float(np.sum(values * stats["support"]) / total)
    if average == "binary":
        positive = np.where(stats["labels"] == 1)[0]
        if positive.size != 1:
            raise ValueError(
                "average='binary' 要求标签中包含正类 1；"
                "可显式传入 labels=[0, 1]"
            )
        return float(values[positive[0]])

    # micro：先汇总 TP / FP / FN，再计算指标
    tp = stats["tp"].sum()
    fp = stats["fp"].sum()
    fn = stats["fn"].sum()
    if metric == "precision":
        return float(_safe_div(tp, tp + fp, zero_division))
    if metric == "recall":
        return float(_safe_div(tp, tp + fn, zero_division))

    precision = float(_safe_div(tp, tp + fp, zero_division))
    recall = float(_safe_div(tp, tp + fn, zero_division))
    return float(_safe_div(2.0 * precision * recall, precision + recall, zero_division))


def _compute(metric, y_true, y_pred, average, labels, zero_division):
    stats = _per_class(y_true, y_pred, labels, zero_division)
    return _aggregate(metric, stats, average, zero_division)


# ----------------------------------------------------------------------
# 公开的评价指标
# ----------------------------------------------------------------------
def calculate_accuracy(y_true, y_pred):
    """准确率：预测正确的样本数 / 总样本数。

    ``y_pred`` 可以是类别下标，也可以是 ``(N, C)`` 的分数矩阵。

    示例
    ----
    >>> calculate_accuracy([0, 1, 1, 0], [0, 1, 0, 0])
    0.75
    """
    true, pred = _prepare(y_true, y_pred)
    return float(np.mean(true == pred))


def calculate_precision(
    y_true, y_pred, average="macro", labels=None, zero_division=0
):
    """精确率 ``TP / (TP + FP)``。

    参数
    ----
    average : {"macro", "micro", "weighted", "binary", None}
        多分类默认 ``"macro"``；``"binary"`` 返回正类（标签 1）的精确率；
        ``None`` 返回逐类数组。
    labels : array_like, optional
        参与统计的类别及顺序。
    zero_division : {0, 1}
        分母为 0 时的返回值（默认 0）。
    """
    return _compute("precision", y_true, y_pred, average, labels, zero_division)


def calculate_recall(
    y_true, y_pred, average="macro", labels=None, zero_division=0
):
    """召回率 ``TP / (TP + FN)``，参数含义同 :func:`calculate_precision`。"""
    return _compute("recall", y_true, y_pred, average, labels, zero_division)


def calculate_f1(y_true, y_pred, average="macro", labels=None, zero_division=0):
    """F1 分数 ``2PR / (P + R)``，参数含义同 :func:`calculate_precision`。"""
    return _compute("f1", y_true, y_pred, average, labels, zero_division)


def calculate_metrics(
    y_true, y_pred, average="macro", labels=None, zero_division=0
):
    """一次性返回常用指标，方便训练循环记录日志。

    返回
    ----
    dict
        包含 ``accuracy``、``precision``、``recall``、``f1``、``support``。

    示例
    ----
    >>> m = calculate_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    >>> round(m["f1"], 4)
    0.7333
    """
    return {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "precision": calculate_precision(
            y_true, y_pred, average, labels, zero_division
        ),
        "recall": calculate_recall(
            y_true, y_pred, average, labels, zero_division
        ),
        "f1": calculate_f1(y_true, y_pred, average, labels, zero_division),
        "support": int(np.asarray(labels_from_scores(y_true)).reshape(-1).size),
    }


def classification_report(
    y_true, y_pred, labels=None, digits=4, zero_division=0
):
    """生成与 sklearn 风格接近的分类报告字符串。

    报表包含每个类别的 precision / recall / f1-score / support，
    以及 accuracy、macro avg、weighted avg 三行汇总。
    """
    digits = int(digits)
    if digits < 0:
        raise ValueError("digits 不能为负数")

    stats = _per_class(y_true, y_pred, labels, zero_division)
    labels_arr = stats["labels"]
    accuracy = calculate_accuracy(y_true, y_pred)

    width = max(12, max(len(str(label)) for label in labels_arr) + 2)
    lines = [
        f"{'':>{width}}{'precision':>12}{'recall':>10}{'f1-score':>10}{'support':>10}",
        "",
    ]
    for i, label in enumerate(labels_arr):
        lines.append(
            f"{str(label):>{width}}"
            f"{stats['precision'][i]:>12.{digits}f}"
            f"{stats['recall'][i]:>10.{digits}f}"
            f"{stats['f1'][i]:>10.{digits}f}"
            f"{int(stats['support'][i]):>10d}"
        )

    total = int(stats["support"].sum())
    macro_p = np.mean(stats["precision"])
    macro_r = np.mean(stats["recall"])
    macro_f1 = np.mean(stats["f1"])
    weighted_p = np.sum(stats["precision"] * stats["support"]) / total
    weighted_r = np.sum(stats["recall"] * stats["support"]) / total
    weighted_f1 = np.sum(stats["f1"] * stats["support"]) / total

    lines += [
        "",
        f"{'accuracy':>{width}}{accuracy:>32.{digits}f}{total:>10d}",
        f"{'macro avg':>{width}}"
        f"{macro_p:>12.{digits}f}{macro_r:>10.{digits}f}"
        f"{macro_f1:>10.{digits}f}{total:>10d}",
        f"{'weighted avg':>{width}}"
        f"{weighted_p:>12.{digits}f}{weighted_r:>10.{digits}f}"
        f"{weighted_f1:>10.{digits}f}{total:>10d}",
    ]
    return "\n".join(lines)
