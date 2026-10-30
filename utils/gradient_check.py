"""数值梯度检查工具（成员 4：损失函数与评价指标）。

手写反向传播时最容易出错的是梯度公式（符号、转置、除以 batch size 等）。
本模块用**中心差分**近似数值梯度，并与解析梯度逐元素比较，
用于验证损失函数、激活函数、全连接层、卷积层等的 ``backward()`` 是否正确。

典型用法::

    from utils.gradient_check import check_gradient

    def loss_fn(w):
        logits = x @ w
        return ce.forward(logits, y)

    result = check_gradient(loss_fn, w, analytic_grad=ce.backward() @ x.T)
    print(result["max_rel_error"])

注意
----
* 被检查的函数必须是“数组 → 标量”，且内部使用 ``float64`` 计算，
  否则差分误差会被浮点噪声淹没。
* 中心差分的步长 ``eps`` 默认 ``1e-6``；对 ReLU 等不可导点附近
  的检查没有意义。
* 函数 ``f`` 不应修改传入的数组，也不应缓存上一次的输入。
"""

import numpy as np

__all__ = ["numerical_gradient", "relative_error", "check_gradient"]


def numerical_gradient(f, x, eps=1e-6):
    """用中心差分计算标量函数 ``f`` 在 ``x`` 处的数值梯度。

    参数
    ----
    f : callable
        ``f(x) -> float``，输入是与 ``x`` 同形状的 ``np.ndarray``。
    x : array_like
        求梯度的位置。
    eps : float
        差分步长。

    返回
    ----
    np.ndarray
        与 ``x`` 同形状的数值梯度。
    """
    if eps <= 0:
        raise ValueError("eps 必须大于 0")

    x = np.array(x, dtype=np.float64, copy=True)
    if x.size == 0:
        raise ValueError("x 不能为空")

    flat = x.reshape(-1)
    grad = np.zeros_like(flat)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = float(f(x))
        flat[i] = original - eps
        minus = float(f(x))
        flat[i] = original
        grad[i] = (plus - minus) / (2.0 * eps)

    return grad.reshape(x.shape)


def relative_error(analytic, numerical, eps=1e-12):
    """计算相对误差 ``|a - n| / max(|a| + |n|, eps)``，逐元素返回。"""
    analytic = np.asarray(analytic, dtype=np.float64)
    numerical = np.asarray(numerical, dtype=np.float64)
    denominator = np.maximum(np.abs(analytic) + np.abs(numerical), eps)
    return np.abs(analytic - numerical) / denominator


def check_gradient(
    f,
    x,
    analytic_grad,
    eps=1e-6,
    rtol=1e-4,
    atol=1e-6,
    raise_error=True,
):
    """比较解析梯度与数值梯度。

    参数
    ----
    f : callable
        标量函数 ``f(x) -> float``。
    x : array_like
        函数输入。
    analytic_grad : array_like
        手写 ``backward()`` 得到的梯度。
    eps, rtol, atol : float
        差分步长与相对/绝对容差。
    raise_error : bool
        不通过时是否抛出 :class:`AssertionError`（默认 ``True``）。

    返回
    ----
    dict
        包含 ``ok``、``max_abs_error``、``max_rel_error``、
        ``worst_index`` 与 ``shape``。

    示例
    ----
    >>> import numpy as np
    >>> result = check_gradient(
    ...     lambda w: float(np.sum(w ** 2)),
    ...     np.array([1.0, -2.0]),
    ...     2.0 * np.array([1.0, -2.0]),
    ... )
    >>> result["ok"]
    True
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("x 不能为空")

    numerical = numerical_gradient(f, x, eps)
    analytic = np.asarray(analytic_grad, dtype=np.float64)
    if analytic.shape != numerical.shape:
        raise ValueError(
            f"解析梯度形状 {analytic.shape} 与输入形状 {numerical.shape} 不一致"
        )

    abs_error = np.abs(analytic - numerical)
    rel_error = relative_error(analytic, numerical)
    max_abs = float(np.max(abs_error))
    max_rel = float(np.max(rel_error))
    worst = np.unravel_index(int(np.argmax(abs_error)), abs_error.shape)
    ok = bool(np.allclose(analytic, numerical, rtol=rtol, atol=atol))

    if not ok and raise_error:
        raise AssertionError(
            "梯度检查未通过：解析梯度与数值梯度不一致。\n"
            f"  最大绝对误差 {max_abs:.3e}（位置 {tuple(int(i) for i in worst)}）\n"
            f"  最大相对误差 {max_rel:.3e}\n"
            f"  解析值 {analytic[worst]!r}，数值值 {numerical[worst]!r}\n"
            f"  容差 rtol={rtol}, atol={atol}"
        )

    return {
        "ok": ok,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "worst_index": tuple(int(i) for i in worst),
        "shape": analytic.shape,
    }
