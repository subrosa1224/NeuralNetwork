import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.losses import CrossEntropyLoss, MSELoss, build_loss
from utils.gradient_check import check_gradient


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
    except Exception as other:
        raise AssertionError(
            f"期望抛出 {exc_type.__name__}，实际抛出 "
            f"{type(other).__name__}: {other}"
        )
    raise AssertionError(f"期望抛出 {exc_type.__name__}，但没有抛出任何异常")


def test_mse_known_values():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.zeros_like(pred)

    mean_loss = MSELoss()
    assert mean_loss.forward(pred, target) == 7.5
    assert_close(mean_loss.backward(), pred / 2.0, message="MSE mean 梯度")

    sum_loss = MSELoss(reduction="sum")
    assert sum_loss.forward(pred, target) == 30.0
    assert_close(sum_loss.backward(), 2.0 * pred, message="MSE sum 梯度")

    none_loss = MSELoss(reduction="none")
    assert_close(
        none_loss.forward(pred, target),
        [[1.0, 4.0], [9.0, 16.0]],
        message="MSE none 损失",
    )
    assert_close(none_loss.backward(), 2.0 * pred, message="MSE none 梯度")
    assert_close(
        none_loss.backward(np.full_like(pred, 0.5)),
        pred,
        message="MSE none 上游梯度",
    )

def test_mse_zero_loss():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(3, 4))
    loss = MSELoss()
    assert loss.forward(pred, pred.copy()) == 0.0
    assert_close(loss.backward(), np.zeros_like(pred))

def test_mse_grad_output_scaling():
    rng = np.random.default_rng(1)
    pred = rng.normal(size=(2, 3))
    target = rng.normal(size=(2, 3))
    loss = MSELoss()
    loss.forward(pred, target)
    base = loss.backward()
    assert_close(loss.backward(2.0), 2.0 * base)
    assert_raises(ValueError, loss.backward, np.ones((2, 3)))

def test_mse_gradient_matches_numerical():
    rng = np.random.default_rng(2)
    pred = rng.normal(size=(4, 3))
    target = rng.normal(size=(4, 3))

    for reduction in ("mean", "sum"):
        loss = MSELoss(reduction=reduction)
        loss.forward(pred, target)
        analytic = loss.backward()
        result = check_gradient(
            lambda p: loss.forward(p, target), pred, analytic
        )
        assert result["ok"], f"MSE({reduction}) 梯度检查失败"

def test_mse_shape_mismatch():
    loss = MSELoss()
    assert_raises(
        ValueError, loss.forward, np.zeros((2, 3)), np.zeros((2, 4))
    )


def test_ce_known_value():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    labels = np.array([0, 1])
    expected = -np.log(np.exp(2.0) / (np.exp(2.0) + np.exp(1.0) + np.exp(0.0)))

    loss = CrossEntropyLoss()
    value = loss.forward(logits, labels)
    assert_close(value, expected, tol=1e-12)
    assert abs(value - 0.40760596444438046) < 1e-12

def test_ce_uniform_logits():
    logits = np.zeros((3, 3))
    labels = np.array([0, 1, 2])
    loss = CrossEntropyLoss()
    assert_close(loss.forward(logits, labels), np.log(3.0))

    expected_grad = np.full((3, 3), 1.0 / 3.0)
    expected_grad[np.arange(3), labels] -= 1.0
    assert_close(loss.backward(), expected_grad / 3.0)

def test_ce_soft_targets_known_value():
    logits = np.zeros((1, 3))
    soft = np.array([[0.5, 0.5, 0.0]])
    loss = CrossEntropyLoss()
    assert_close(loss.forward(logits, soft), np.log(3.0))
    assert_close(
        loss.backward(),
        [[1.0 / 3.0 - 0.5, 1.0 / 3.0 - 0.5, 1.0 / 3.0]],
    )

    # 行和不为 1 时，梯度公式仍为 p * sum(t) - t
    soft_unnormalized = np.array([[0.2, 0.3, 0.1]])
    loss.forward(logits, soft_unnormalized)
    expected = np.full((1, 3), 1.0 / 3.0) * 0.6 - soft_unnormalized
    assert_close(loss.backward(), expected)

def test_ce_single_sample():
    logits = np.array([1.0, 2.0, 3.0])
    loss = CrossEntropyLoss()
    value = loss.forward(logits, 2)
    expected = -(3.0 - np.log(np.exp(1.0) + np.exp(2.0) + np.exp(3.0)))
    assert_close(value, expected)

    grad = loss.backward()
    assert grad.shape == (3,)
    assert_close(np.sum(grad), 0.0, tol=1e-12)

    none_loss = CrossEntropyLoss(reduction="none")
    none_value = none_loss.forward(logits, 2)
    assert np.isscalar(none_value) or np.asarray(none_value).shape == ()
    assert none_loss.backward().shape == (3,)

def test_ce_numerically_stable():
    logits = np.array([[1000.0, 999.0, 998.0]])
    loss = CrossEntropyLoss()
    value = loss.forward(logits, np.array([0]))
    assert np.isfinite(value)
    assert 0.4 < value < 0.42  # log(1 + e^-1 + e^-2)

    grad = loss.backward()
    assert np.all(np.isfinite(grad))
    assert_close(np.sum(grad), 0.0, tol=1e-12)

    # logits 差距极大时 softmax 会下溢为 0，但不应该产生 nan
    extreme = np.array([[1000.0, -1000.0, -1000.0]])
    extreme_loss = CrossEntropyLoss()
    assert extreme_loss.forward(extreme, np.array([0])) == 0.0
    assert_close(extreme_loss.backward(), [[0.0, 0.0, 0.0]])

    # 输入概率包含 0 时也不能出现 nan
    prob_loss = CrossEntropyLoss(from_logits=False)
    value = prob_loss.forward(np.array([[0.0, 1.0]]), np.array([0]))
    assert np.isfinite(value)
    assert np.all(np.isfinite(prob_loss.backward()))

def test_ce_from_probabilities():
    probs = np.array([[0.5, 0.5, 0.0]])
    labels = np.array([0])
    loss = CrossEntropyLoss(from_logits=False)
    assert_close(loss.forward(probs, labels), np.log(2.0))
    assert_close(loss.backward(), [[-2.0, 0.0, 0.0]])

    single = np.array([0.8, 0.2])
    assert_close(loss.forward(single, 1), -np.log(0.2))
    assert_close(loss.backward(), [0.0, -5.0])

def test_ce_from_probabilities_soft_targets():
    probs = np.array([[0.5, 0.5, 0.0]])
    soft = np.array([[0.5, 0.5, 0.0]])
    loss = CrossEntropyLoss(from_logits=False)
    assert_close(loss.forward(probs, soft), np.log(2.0))
    assert_close(loss.backward(), [[-1.0, -1.0, 0.0]])

def test_ce_one_hot_equals_labels():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(6, 5))
    labels = rng.integers(0, 5, size=6)
    one_hot = np.zeros((6, 5))
    one_hot[np.arange(6), labels] = 1.0

    loss_a = CrossEntropyLoss()
    loss_b = CrossEntropyLoss()
    assert_close(
        loss_a.forward(logits, labels), loss_b.forward(logits, one_hot)
    )
    assert_close(loss_a.backward(), loss_b.backward())


def test_ce_hard_label_gradient_matches_numerical():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(5, 4))
    labels = rng.integers(0, 4, size=5)

    loss = CrossEntropyLoss()
    loss.forward(logits, labels)
    analytic = loss.backward()
    result = check_gradient(
        lambda z: loss.forward(z, labels), logits, analytic
    )
    assert result["ok"]
    assert result["max_rel_error"] < 1e-6

def test_ce_soft_target_gradient_matches_numerical():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(4, 3))
    raw = rng.random((4, 3))
    soft = raw / raw.sum(axis=1, keepdims=True)

    loss = CrossEntropyLoss()
    loss.forward(logits, soft)
    result = check_gradient(
        lambda z: loss.forward(z, soft), logits, loss.backward()
    )
    assert result["ok"]

def test_ce_probability_gradient_matches_numerical():
    rng = np.random.default_rng(6)
    raw = rng.random((3, 4)) + 0.2
    probs = raw / raw.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 4, size=3)

    loss = CrossEntropyLoss(from_logits=False)
    loss.forward(probs, labels)
    result = check_gradient(
        lambda p: loss.forward(p, labels), probs, loss.backward()
    )
    assert result["ok"]

    soft = np.full((3, 4), 1.0 / 4.0)
    loss_soft = CrossEntropyLoss(from_logits=False)
    loss_soft.forward(probs, soft)
    result = check_gradient(
        lambda p: loss_soft.forward(p, soft), probs, loss_soft.backward()
    )
    assert result["ok"]

def test_ce_reductions():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(3, 2))
    labels = rng.integers(0, 2, size=3)

    mean_loss = CrossEntropyLoss()
    sum_loss = CrossEntropyLoss(reduction="sum")
    none_loss = CrossEntropyLoss(reduction="none")

    mean_value = mean_loss.forward(logits, labels)
    sum_value = sum_loss.forward(logits, labels)
    per_sample = none_loss.forward(logits, labels)

    assert_close(sum_value, mean_value * 3)
    assert_close(np.sum(per_sample), sum_value)
    assert_close(np.mean(per_sample), mean_value)

    assert_close(mean_loss.backward() * 3.0, sum_loss.backward())
    assert_close(
        none_loss.backward(np.ones(3)), mean_loss.backward() * 3.0
    )

def test_ce_large_batch_mean_divides_by_n():
    logits = np.zeros((4, 10))
    labels = np.array([0, 1, 2, 3])
    loss = CrossEntropyLoss()
    assert_close(loss.forward(logits, labels), np.log(10.0))
    grad = loss.backward()
    assert_close(np.sum(grad, axis=1), np.zeros(4), tol=1e-12)
    assert_close(np.sum(grad), 0.0, tol=1e-12)


def test_backward_before_forward_raises():
    assert_raises(RuntimeError, MSELoss().backward)
    assert_raises(RuntimeError, CrossEntropyLoss().backward)

def test_failed_forward_invalidates_cache():
    mse = MSELoss()
    mse.forward(np.ones((2, 2)), np.zeros((2, 2)))
    assert_raises(
        ValueError, mse.forward, np.ones((2, 3)), np.zeros((2, 2))
    )
    assert mse.last_loss is None
    assert_raises(RuntimeError, mse.backward)

    ce = CrossEntropyLoss()
    ce.forward(np.zeros((2, 3)), np.array([0, 1]))
    assert_raises(
        ValueError, ce.forward, np.zeros((2, 3)), np.array([0, 5])
    )
    assert_raises(RuntimeError, ce.backward)

def test_forward_twice_uses_latest_inputs():
    rng = np.random.default_rng(8)
    logits_a = rng.normal(size=(2, 3))
    logits_b = rng.normal(size=(2, 3))
    labels = np.array([0, 2])

    loss = CrossEntropyLoss()
    loss.forward(logits_a, labels)
    loss.forward(logits_b, labels)
    fresh = CrossEntropyLoss()
    fresh.forward(logits_b, labels)
    assert_close(loss.backward(), fresh.backward())

def test_ce_invalid_targets():
    logits = np.zeros((2, 3))
    loss = CrossEntropyLoss()

    assert_raises(ValueError, loss.forward, logits, np.array([0, 3]))
    assert_raises(ValueError, loss.forward, logits, np.array([-1, 0]))
    assert_raises(ValueError, loss.forward, logits, np.array([0.5, 1.0]))
    assert_raises(ValueError, loss.forward, logits, np.array([0]))
    assert_raises(ValueError, loss.forward, logits, np.zeros((2, 2)))
    assert_raises(
        ValueError, loss.forward, logits, np.array([[-0.1, 0.6, 0.5], [1, 0, 0]])
    )

def test_ce_invalid_pred():
    loss = CrossEntropyLoss()
    assert_raises(ValueError, loss.forward, np.zeros((0, 3)), np.zeros(0))
    assert_raises(ValueError, loss.forward, np.zeros((2, 3, 4)), np.zeros(2))

def test_constructor_validation():
    assert_raises(ValueError, MSELoss, "avg")
    assert_raises(ValueError, CrossEntropyLoss, "avg")
    assert_raises(ValueError, CrossEntropyLoss, "mean", True, 0.0)

def test_build_loss():
    assert isinstance(build_loss("mse"), MSELoss)
    assert isinstance(build_loss("MSE", reduction="sum"), MSELoss)
    assert isinstance(build_loss("cross-entropy"), CrossEntropyLoss)
    assert isinstance(build_loss("CE", from_logits=False), CrossEntropyLoss)
    assert build_loss("mse", reduction="sum").reduction == "sum"
    assert_raises(ValueError, build_loss, "huber")

def test_call_operator_and_last_loss():
    loss = MSELoss()
    value = loss(np.ones((2, 2)), np.zeros((2, 2)))
    assert value == 1.0
    assert loss.last_loss == 1.0
    assert "MSELoss" in repr(loss)
