import numpy as np


def numerical_gradient(f, x, eps=1e-6):
    if eps <= 0:
        raise ValueError("eps 必须大于 0")

    x = np.array(x, dtype=float, copy=True)
    if x.size == 0:
        raise ValueError("x 不能为空")

    flat = x.reshape(-1)
    grad = np.zeros_like(flat)
    for i in range(flat.size):
        # 中心差分
        old = flat[i]
        flat[i] = old + eps
        plus = float(f(x))
        flat[i] = old - eps
        minus = float(f(x))
        flat[i] = old
        grad[i] = (plus - minus) / (2.0 * eps)
    return grad.reshape(x.shape)


def relative_error(analytic, numerical, eps=1e-12):
    analytic = np.asarray(analytic, dtype=float)
    numerical = np.asarray(numerical, dtype=float)
    denom = np.maximum(np.abs(analytic) + np.abs(numerical), eps)
    return np.abs(analytic - numerical) / denom


def check_gradient(
    f,
    x,
    analytic_grad,
    eps=1e-6,
    rtol=1e-4,
    atol=1e-6,
    raise_error=True,
):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x 不能为空")

    numerical = numerical_gradient(f, x, eps)
    analytic = np.asarray(analytic_grad, dtype=float)
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
            f"梯度检查未通过：最大绝对误差 {max_abs:.3e}，"
            f"最大相对误差 {max_rel:.3e}，最差位置 {tuple(int(i) for i in worst)}"
        )

    return {
        "ok": ok,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "worst_index": tuple(int(i) for i in worst),
        "shape": analytic.shape,
    }
