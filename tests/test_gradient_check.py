"""数值梯度检查工具单元测试（成员 4）。

运行方式（在项目根目录）：python run_tests.py
或：pytest tests/test_gradient_check.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.gradient_check import check_gradient, numerical_gradient, relative_error


def assert_close(actual, expected, tol=1e-9, message=""):
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


def test_numerical_gradient_quadratic():
    """f(x) = sum(x^2) 的数值梯度应为 2x。"""
    x = np.array([[1.0, -2.0], [0.5, 3.0]])
    grad = numerical_gradient(lambda v: float(np.sum(v ** 2)), x)
    assert_close(grad, 2.0 * x, tol=1e-8)


def test_check_gradient_passes():
    x = np.array([1.0, -2.0, 0.3])
    result = check_gradient(
        lambda w: float(np.sum(w ** 2)), x, analytic_grad=2.0 * x
    )
    assert result["ok"] is True
    assert result["max_rel_error"] < 1e-7
    assert result["shape"] == x.shape


def test_check_gradient_detects_wrong_gradient():
    """错误的解析梯度必须被识别出来。"""
    x = np.array([1.0, 2.0, 3.0])
    wrong = np.array([0.0, 0.0, 0.0])

    result = check_gradient(
        lambda w: float(np.sum(w ** 2)), x, wrong, raise_error=False
    )
    assert result["ok"] is False
    assert result["max_abs_error"] > 1.0

    assert_raises(
        AssertionError,
        check_gradient,
        lambda w: float(np.sum(w ** 2)),
        x,
        wrong,
    )


def test_check_gradient_shape_mismatch():
    x = np.zeros((2, 3))
    assert_raises(
        ValueError,
        check_gradient,
        lambda w: float(np.sum(w ** 2)),
        x,
        np.zeros(6),
    )


def test_numerical_gradient_does_not_modify_input():
    x = np.array([1.0, 2.0])
    original = x.copy()
    numerical_gradient(lambda v: float(np.sum(v ** 2)), x)
    assert np.array_equal(x, original)


def test_relative_error_formula():
    analytic = np.array([1.0, 2.0, 0.0])
    numerical = np.array([1.0, 2.0, 1e-3])
    error = relative_error(analytic, numerical)
    assert error[0] == 0.0
    assert error[1] == 0.0
    assert error[2] > 0.9  # 0 到 1e-3 的相对误差接近 1


def test_negative_eps_rejected():
    assert_raises(
        ValueError, numerical_gradient, lambda v: float(np.sum(v)), np.ones(2), 0.0
    )
