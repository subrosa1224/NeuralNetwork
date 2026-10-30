"""评价指标单元测试（成员 4）。

运行方式（在项目根目录）：python run_tests.py
或：pytest tests/test_metrics.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.metrics import (
    calculate_accuracy,
    calculate_f1,
    calculate_metrics,
    calculate_precision,
    calculate_recall,
    classification_report,
    confusion_matrix,
    labels_from_scores,
)


# ----------------------------------------------------------------------
# 测试辅助函数
# ----------------------------------------------------------------------
def assert_close(actual, expected, tol=1e-10, message=""):
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    assert actual_arr.shape == expected_arr.shape, (
        f"{message} 形状不一致：{actual_arr.shape} != {expected_arr.shape}"
    )
    if not np.allclose(actual_arr, expected_arr, rtol=0.0, atol=tol):
        raise AssertionError(
            f"{message} 数值不一致：\n实际 {actual_arr}\n期望 {expected_arr}"
        )


def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as other:  # noqa: BLE001
        raise AssertionError(
            f"期望抛出 {exc_type.__name__}，实际抛出 "
            f"{type(other).__name__}: {other}"
        )
    raise AssertionError(f"期望抛出 {exc_type.__name__}，但没有抛出任何异常")


# 固定的多分类样例：混淆矩阵为
#   [[3, 0, 0],
#    [0, 2, 2],
#    [1, 1, 1]]
MULTI_TRUE = np.array([0, 1, 2, 2, 1, 0, 1, 2, 0, 1])
MULTI_PRED = np.array([0, 2, 2, 1, 1, 0, 1, 0, 0, 2])

# 固定的二分类样例：TP=2, FP=1, FN=2, TN=3, accuracy=5/8
BIN_TRUE = np.array([1, 1, 1, 1, 0, 0, 0, 0])
BIN_PRED = np.array([1, 1, 0, 0, 1, 0, 0, 0])


# ----------------------------------------------------------------------
# 准确率与混淆矩阵
# ----------------------------------------------------------------------
def test_accuracy_binary():
    assert_close(calculate_accuracy(BIN_TRUE, BIN_PRED), 5.0 / 8.0)
    assert_close(calculate_accuracy([0, 1, 1, 0], [0, 1, 0, 0]), 0.75)


def test_accuracy_with_scores():
    """y_pred 为分数矩阵时按行 argmax。"""
    y_true = np.array([0, 1, 2])
    scores = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
    )
    assert_close(calculate_accuracy(y_true, scores), 1.0)


def test_confusion_matrix_layout():
    """混淆矩阵的行是真实类别、列是预测类别。"""
    cm = confusion_matrix(MULTI_TRUE, MULTI_PRED)
    expected = np.array([[3, 0, 0], [0, 2, 2], [1, 1, 1]])
    assert np.array_equal(cm, expected)

    binary_cm = confusion_matrix(BIN_TRUE, BIN_PRED)
    assert np.array_equal(binary_cm, np.array([[3, 1], [2, 2]]))


def test_confusion_matrix_labels_order():
    """labels 决定行列顺序，且允许自定义顺序。"""
    y_true = np.array([2, 2, 1, 0])
    y_pred = np.array([2, 1, 1, 0])
    cm = confusion_matrix(y_true, y_pred, labels=[2, 1, 0])
    assert np.array_equal(cm, np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]))

    default_cm = confusion_matrix(y_true, y_pred)
    assert np.array_equal(default_cm, np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]]))


# ----------------------------------------------------------------------
# 精确率 / 召回率 / F1
# ----------------------------------------------------------------------
def test_binary_precision_recall_f1():
    assert_close(
        calculate_precision(BIN_TRUE, BIN_PRED, average="binary"), 2.0 / 3.0
    )
    assert_close(
        calculate_recall(BIN_TRUE, BIN_PRED, average="binary"), 0.5
    )
    assert_close(calculate_f1(BIN_TRUE, BIN_PRED, average="binary"), 4.0 / 7.0)


def test_multiclass_per_class_values():
    expected_precision = [0.75, 2.0 / 3.0, 1.0 / 3.0]
    expected_recall = [1.0, 0.5, 1.0 / 3.0]
    expected_f1 = [6.0 / 7.0, 4.0 / 7.0, 1.0 / 3.0]

    assert_close(
        calculate_precision(MULTI_TRUE, MULTI_PRED, average=None),
        expected_precision,
    )
    assert_close(
        calculate_recall(MULTI_TRUE, MULTI_PRED, average=None),
        expected_recall,
    )
    assert_close(
        calculate_f1(MULTI_TRUE, MULTI_PRED, average=None), expected_f1
    )


def test_multiclass_averages():
    """macro / weighted / micro 三种平均与手算一致。"""
    assert_close(
        calculate_precision(MULTI_TRUE, MULTI_PRED, average="macro"), 7.0 / 12.0
    )
    assert_close(
        calculate_recall(MULTI_TRUE, MULTI_PRED, average="macro"), 11.0 / 18.0
    )
    assert_close(
        calculate_f1(MULTI_TRUE, MULTI_PRED, average="macro"), 37.0 / 63.0
    )

    assert_close(
        calculate_precision(MULTI_TRUE, MULTI_PRED, average="weighted"),
        71.0 / 120.0,
    )
    assert_close(
        calculate_recall(MULTI_TRUE, MULTI_PRED, average="weighted"), 0.6
    )
    assert_close(
        calculate_f1(MULTI_TRUE, MULTI_PRED, average="weighted"), 41.0 / 70.0
    )

    for fn in (calculate_precision, calculate_recall, calculate_f1):
        assert_close(
            fn(MULTI_TRUE, MULTI_PRED, average="micro"), 0.6
        )


def test_perfect_predictions():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=100)
    assert_close(calculate_accuracy(labels, labels), 1.0)
    for fn in (calculate_precision, calculate_recall, calculate_f1):
        assert_close(fn(labels, labels), 1.0)
        assert_close(fn(labels, labels, average="micro"), 1.0)
        assert_close(fn(labels, labels, average="weighted"), 1.0)


def test_zero_division_handling():
    """某一类没有样本 / 预测时，按 zero_division 处理。"""
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    labels = [0, 1]

    assert_close(
        calculate_precision(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        ),
        [0.0, 1.0],
    )
    assert_close(
        calculate_precision(
            y_true, y_pred, average=None, labels=labels, zero_division=1
        ),
        [1.0, 1.0],
    )
    assert_close(
        calculate_f1(y_true, y_pred, labels=labels, zero_division=1), 1.0
    )

    # 数据中只有类别 0 时，binary 平均无法确定正类
    assert_raises(
        ValueError,
        calculate_precision,
        np.array([0, 0]),
        np.array([0, 0]),
        average="binary",
    )


def test_one_hot_true_labels():
    """y_true 为 one-hot 时按行 argmax。"""
    y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    y_pred = np.array([1, 0, 2])
    assert_close(calculate_accuracy(y_true, y_pred), 1.0)


def test_scores_equivalent_to_labels():
    """分数矩阵输入与先 argmax 再计算的结果完全一致。"""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 4, size=50)
    scores = rng.normal(size=(50, 4))
    y_pred = np.argmax(scores, axis=1)

    assert_close(
        calculate_accuracy(y_true, scores), calculate_accuracy(y_true, y_pred)
    )
    for fn in (calculate_precision, calculate_recall, calculate_f1):
        assert_close(fn(y_true, scores), fn(y_true, y_pred))
        assert_close(
            fn(y_true, scores, average=None), fn(y_true, y_pred, average=None)
        )


def test_micro_f1_equals_accuracy():
    """单标签多分类中，micro-F1 与准确率相等。"""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 5, size=200)
    y_pred = rng.integers(0, 5, size=200)
    assert_close(
        calculate_f1(y_true, y_pred, average="micro"),
        calculate_accuracy(y_true, y_pred),
    )


def test_weighted_equals_macro_when_balanced():
    """各类 support 相同时，weighted 平均等于 macro 平均。"""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    for fn in (calculate_precision, calculate_recall, calculate_f1):
        assert_close(
            fn(y_true, y_pred, average="macro"),
            fn(y_true, y_pred, average="weighted"),
        )


def test_f1_is_harmonic_mean():
    """逐类 F1 应等于 P、R 的调和平均。"""
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 2, 1, 1, 2, 0, 2])
    precision = calculate_precision(y_true, y_pred, average=None)
    recall = calculate_recall(y_true, y_pred, average=None)
    expected = 2 * precision * recall / (precision + recall)
    assert_close(calculate_f1(y_true, y_pred, average=None), expected)


# ----------------------------------------------------------------------
# 汇总接口与报告
# ----------------------------------------------------------------------
def test_metrics_dict():
    metrics = calculate_metrics(MULTI_TRUE, MULTI_PRED)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "support"}
    assert metrics["support"] == 10
    assert_close(metrics["accuracy"], 0.6)
    assert_close(metrics["precision"], calculate_precision(MULTI_TRUE, MULTI_PRED))
    assert_close(metrics["recall"], calculate_recall(MULTI_TRUE, MULTI_PRED))
    assert_close(metrics["f1"], calculate_f1(MULTI_TRUE, MULTI_PRED))


def test_classification_report():
    report = classification_report(MULTI_TRUE, MULTI_PRED, digits=4)
    assert isinstance(report, str)
    for keyword in ("precision", "recall", "f1-score", "support"):
        assert keyword in report
    for keyword in ("accuracy", "macro avg", "weighted avg"):
        assert keyword in report
    assert "0.6000" in report
    assert "0.5873" in report
    assert len(report.splitlines()) >= 8


def test_labels_from_scores():
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    assert np.array_equal(labels_from_scores(scores), [1, 0, 1])
    assert np.array_equal(labels_from_scores([2, 0, 1]), [2, 0, 1])
    assert_raises(ValueError, labels_from_scores, np.zeros((2, 2, 2)))


# ----------------------------------------------------------------------
# 非法输入
# ----------------------------------------------------------------------
def test_invalid_inputs():
    assert_raises(
        ValueError, calculate_accuracy, [0, 1], [0]
    )
    assert_raises(ValueError, calculate_accuracy, [], [])
    assert_raises(
        ValueError, calculate_accuracy, np.zeros((2, 2, 2)), np.zeros(2)
    )
    assert_raises(
        ValueError, calculate_accuracy, [0.5, 1.0], [0, 1]
    )
    assert_raises(
        ValueError, calculate_accuracy, ["a", "b"], ["a", "b"]
    )
    assert_raises(
        ValueError, confusion_matrix, [0, 1], [0, 1], labels=[0, 1, 1]
    )
    assert_raises(
        ValueError, confusion_matrix, [0, 1], [0, 1], labels=[0]
    )


def test_invalid_average():
    for average in ("macro ", "avg", 1):
        assert_raises(
            ValueError,
            calculate_precision,
            MULTI_TRUE,
            MULTI_PRED,
            average=average,
        )
