import numpy as np

_AVERAGES = ("binary", "macro", "micro", "weighted")


def labels_from_scores(y):
    arr = np.asarray(y)
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError("输入不能为空")
        return np.argmax(arr, axis=1)
    if arr.ndim == 1:
        return arr
    raise ValueError(f"只支持一维标签或二维分数矩阵，收到形状 {arr.shape}")


def _validate_int_values(values, name):
    if values.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} 中不能包含 inf / nan")
    if np.any(values != np.floor(values)):
        raise ValueError(f"{name} 中的类别标签必须是整数")
    return values.astype(np.int64)


def _as_int_labels(y, name):
    arr = np.asarray(y)

    # (N, 1) 的整数列按标签处理
    if arr.ndim == 2 and arr.shape[1] == 1:
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{name} 中的类别标签必须是数值")
        values = arr.astype(float).reshape(-1)
        if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError(
                f"{name} 的形状为 (N, 1) 但不是整数标签列；"
                "二分类概率请先转换为 0/1 标签，或传入 (N, 2) 的概率矩阵"
            )
        return _validate_int_values(values, name)

    arr = labels_from_scores(arr).reshape(-1)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} 中的类别标签必须是数值")
    return _validate_int_values(arr.astype(float), name)


def _prepare(y_true, y_pred):
    true = _as_int_labels(y_true, "y_true")
    pred = _as_int_labels(y_pred, "y_pred")
    if true.size != pred.size:
        raise ValueError(
            f"y_true 与 y_pred 的样本数必须相同，收到 {true.size} 和 {pred.size}"
        )
    return true, pred


def _resolve_labels(labels, true, pred):
    observed = np.unique(np.concatenate((true, pred)))
    if labels is None:
        return observed

    arr = np.asarray(labels)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("labels 必须是长度为 C 的一维数组")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("labels 中的类别标签必须是数值")
    values = arr.astype(float)
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
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.zeros(np.broadcast(num, den).shape, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    if np.any(den == 0):
        out = np.where(den == 0, float(zero_division), out)
    return out


def confusion_matrix(y_true, y_pred, labels=None):
    # 行是真实类别，列是预测类别
    cm, _ = _confusion(y_true, y_pred, labels)
    return cm


def _confusion(y_true, y_pred, labels=None):
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
    cm, labels_arr = _confusion(y_true, y_pred, labels)

    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
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

    # micro：先汇总所有类别的 TP / FP / FN
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


def calculate_accuracy(y_true, y_pred):
    true, pred = _prepare(y_true, y_pred)
    return float(np.mean(true == pred))


def calculate_precision(y_true, y_pred, average="macro", labels=None, zero_division=0):
    return _compute("precision", y_true, y_pred, average, labels, zero_division)


def calculate_recall(y_true, y_pred, average="macro", labels=None, zero_division=0):
    return _compute("recall", y_true, y_pred, average, labels, zero_division)


def calculate_f1(y_true, y_pred, average="macro", labels=None, zero_division=0):
    return _compute("f1", y_true, y_pred, average, labels, zero_division)


def calculate_metrics(y_true, y_pred, average="macro", labels=None, zero_division=0):
    return {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "precision": calculate_precision(
            y_true, y_pred, average, labels, zero_division
        ),
        "recall": calculate_recall(y_true, y_pred, average, labels, zero_division),
        "f1": calculate_f1(y_true, y_pred, average, labels, zero_division),
        "support": int(np.asarray(labels_from_scores(y_true)).reshape(-1).size),
    }


def classification_report(y_true, y_pred, labels=None, digits=4, zero_division=0):
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
