"""损失函数与梯度计算（成员 4：损失函数与评价指标）。

对应最终项目目录中的 ``core/losses.py``。

设计约定
--------
项目中所有层的接口统一为::

    y = layer.forward(x)
    dx = layer.backward(dy)

损失函数沿用同样的风格::

    value = loss.forward(pred, target)   # 标量（reduction="none" 时为数组）
    d_pred = loss.backward()             # 对 pred 的梯度

``pred`` 与 ``target`` 的约定：

* 分类（:class:`CrossEntropyLoss`）：``pred`` 默认是 **未经过 softmax 的
  logits**，形状 ``(N, C)``；``target`` 可以是形如 ``(N,)`` 的类别下标，
  也可以是形如 ``(N, C)`` 的 one-hot / 软标签（每行是概率）。
  单样本时允许 ``pred`` 为 ``(C,)``。
* 回归（:class:`MSELoss`）：``pred`` 与 ``target`` 形状必须完全一致。

数值稳定性
----------
* 交叉熵内部使用 log-sum-exp，logits 很大（例如 1000）时也不会出现
  ``inf`` / ``nan``。
* 当 ``from_logits=False``（输入概率）时，概率会被截断到 ``[eps, +inf)``，
  避免 ``log(0)``。
* 软标签中概率为 0 的位置不会被计入 ``0 * log(0)``。

示例
----
>>> import numpy as np
>>> from core.losses import CrossEntropyLoss
>>> loss = CrossEntropyLoss()
>>> logits = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
>>> loss.forward(logits, np.array([0, 1]))
0.4076059644443804
>>> loss.backward().shape
(2, 3)
"""

import numpy as np

__all__ = ["Loss", "MSELoss", "CrossEntropyLoss", "build_loss"]


def _to_float(x):
    """把输入转成浮点数组；已经是浮点类型时保持原 dtype（不做多余拷贝）。"""
    arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


def _log_softmax(x):
    """按行计算 log(softmax(x))，使用 log-sum-exp 保证数值稳定。"""
    shifted = x - np.max(x, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _validate_labels(labels, n, c):
    """把类别下标整理成 int64 的 ``(n,)`` 数组，并检查取值范围。"""
    arr = np.asarray(labels, dtype=np.float64).reshape(-1)
    if arr.size != n:
        raise ValueError(f"标签数量应为 {n}，收到 {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("标签中不能包含 inf / nan")
    if np.any(arr != np.floor(arr)):
        raise ValueError("类别下标必须是整数")
    arr = arr.astype(np.int64)
    if np.any(arr < 0) or np.any(arr >= c):
        raise ValueError(f"类别下标必须在 [0, {c - 1}] 范围内，收到 {arr}")
    return arr


def _validate_soft_target(target, n, c):
    """把软标签整理成 ``(n, c)`` 的浮点数组。"""
    arr = np.asarray(target, dtype=np.float64)
    if arr.shape != (n, c):
        raise ValueError(f"软标签形状应为 {(n, c)}，收到 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("软标签中不能包含 inf / nan")
    if np.any(arr < 0):
        raise ValueError("软标签中的概率不能为负数")
    return arr


class Loss:
    """损失函数基类，统一 ``reduction`` 与 ``backward()`` 的调用方式。

    参数
    ----
    reduction : {"mean", "sum", "none"}
        * ``"mean"``：对所有（或逐样本）损失取平均，反向时梯度除以元素个数；
        * ``"sum"``：求和；
        * ``"none"``：不做归约，返回逐元素 / 逐样本损失数组。

    属性
    ----
    last_loss : float 或 np.ndarray
        最近一次 ``forward()`` 返回的损失值，便于训练循环记录。
    """

    name = "loss"

    def __init__(self, reduction="mean"):
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction 必须是 'mean'、'sum' 或 'none'，收到 {reduction!r}"
            )
        self.reduction = reduction
        self.last_loss = None
        self._grad = None
        self._loss_count = 1

    # ------------------------------------------------------------------
    # 子类需要实现的接口
    # ------------------------------------------------------------------
    def forward(self, pred, target):
        """计算损失并缓存反向传播所需的中间结果。"""
        raise NotImplementedError

    def backward(self, grad_output=None):
        """返回损失对 ``pred`` 的梯度。

        参数
        ----
        grad_output : None 或 标量/数组
            上游梯度。``reduction="none"`` 时按元素相乘；其余归约方式下
            只接受标量（等价于整体缩放）。
        """
        if self._grad is None:
            raise RuntimeError("backward() 必须在 forward() 之后调用")

        if self.reduction == "none":
            grad = self._grad
            if grad_output is not None:
                grad = grad * np.asarray(grad_output)
            return grad.copy()

        if grad_output is not None:
            scale = np.asarray(grad_output)
            if scale.size != 1:
                raise ValueError(
                    f"reduction={self.reduction!r} 时 grad_output 必须是标量，"
                    f"收到形状 {scale.shape}"
                )
            return self._grad * float(scale.reshape(-1)[0])
        return self._grad.copy()

    def __call__(self, pred, target):
        """允许 ``loss(pred, target)`` 这样调用。"""
        return self.forward(pred, target)

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction={self.reduction!r})"

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _reduce(self, values):
        """按 reduction 汇总损失，并记录平均时的元素个数。"""
        self._loss_count = max(int(values.size), 1)
        if self.reduction == "mean":
            return float(np.mean(values))
        if self.reduction == "sum":
            return float(np.sum(values))
        return values

    def _scale_grad(self, grad):
        """reduction="mean" 时把梯度除以元素个数。"""
        if self.reduction == "mean":
            return grad / self._loss_count
        return grad

    def _invalidate(self):
        """开始新的 forward 时清除上一次的缓存，避免异常后误用旧梯度。"""
        self._grad = None
        self.last_loss = None


class MSELoss(Loss):
    """均方误差（Mean Squared Error）。

    .. math:: L = \\frac{1}{N} \\sum_i (pred_i - target_i)^2

    ``reduction="mean"`` 时对所有元素取平均；反向梯度为
    :math:`\\frac{2}{N}(pred - target)`。

    示例
    ----
    >>> import numpy as np
    >>> loss = MSELoss()
    >>> loss.forward(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    2.5
    >>> loss.backward()
    array([1., 2.])
    """

    name = "mse"

    def forward(self, pred, target):
        self._invalidate()
        pred = _to_float(pred)
        target = _to_float(target)
        if pred.shape != target.shape:
            raise ValueError(
                "MSE 要求 pred 与 target 形状一致，"
                f"收到 {pred.shape} 和 {target.shape}"
            )

        diff = pred - target
        self.last_loss = self._reduce(diff * diff)
        self._grad = self._scale_grad(2.0 * diff)
        return self.last_loss


class CrossEntropyLoss(Loss):
    """交叉熵损失（Cross Entropy），支持 logits 与概率两种输入。

    参数
    ----
    reduction : {"mean", "sum", "none"}
        见 :class:`Loss`。``"mean"`` 除以的是 **样本数 N**。
    from_logits : bool
        * ``True``（默认）：``pred`` 是 logits，内部使用数值稳定的
          log-softmax；
        * ``False``：``pred`` 已经是概率（每行和为 1），内部做 ``log``。
    eps : float
        ``from_logits=False`` 时对概率做截断的下限，防止 ``log(0)``。

    ``target`` 支持两种形式：

    * 类别下标：``(N,)`` 的整数数组（单样本可用标量）；
    * one-hot / 软标签：``(N, C)`` 的数组（单样本可用 ``(C,)``）。

    示例
    ----
    >>> import numpy as np
    >>> loss = CrossEntropyLoss()
    >>> loss.forward(np.zeros((2, 3)), np.array([0, 2]))
    1.0986122886681098
    """

    name = "cross_entropy"

    def __init__(self, reduction="mean", from_logits=True, eps=1e-12):
        super().__init__(reduction)
        self.from_logits = bool(from_logits)
        self.eps = float(eps)
        if self.eps <= 0:
            raise ValueError("eps 必须大于 0")

    def forward(self, pred, target):
        self._invalidate()
        scores = _to_float(pred)
        original_shape = scores.shape

        if scores.ndim == 1:
            if scores.size == 0:
                raise ValueError("pred 不能为空")
            scores = scores.reshape(1, -1)
            single_sample = True
        elif scores.ndim == 2:
            if scores.shape[0] == 0 or scores.shape[1] == 0:
                raise ValueError("pred 不能为空")
            single_sample = False
        else:
            raise ValueError(
                f"CrossEntropyLoss 的 pred 应为 (N, C) 或 (C,)，收到 {scores.shape}"
            )

        n, c = scores.shape
        labels, soft = self._parse_target(target, n, c, single_sample)

        if self.from_logits:
            log_probs = _log_softmax(scores)
            probs = np.exp(log_probs)
        else:
            probs = np.clip(scores, self.eps, None)
            log_probs = np.log(probs)

        rows = np.arange(n)
        if labels is not None:
            # 硬标签：L_i = -log p_{i, y_i}
            values = -log_probs[rows, labels]
            if self.from_logits:
                grad = probs.copy()
                grad[rows, labels] -= 1.0
            else:
                grad = np.zeros_like(probs)
                grad[rows, labels] = -1.0 / probs[rows, labels]
        else:
            # 软标签：L_i = -sum_c t_{i,c} log p_{i,c}
            positive = soft > 0
            target_masked = np.where(positive, soft, 0.0)
            log_masked = np.where(positive, log_probs, 0.0)
            values = -np.sum(target_masked * log_masked, axis=1)

            if self.from_logits:
                # dL/dz = p * sum(t) - t（t 归一化时为 p - t）
                grad = probs * soft.sum(axis=1, keepdims=True) - soft
            else:
                grad = np.where(positive, -soft / probs, 0.0)

        self.last_loss = self._reduce(values)
        if single_sample and self.reduction == "none":
            self.last_loss = float(np.asarray(self.last_loss).reshape(-1)[0])

        self._n = n
        self._single_sample = single_sample
        grad = self._scale_grad(grad)
        self._grad = grad.reshape(original_shape)
        return self.last_loss

    def backward(self, grad_output=None):
        """返回对 ``pred`` 的梯度。

        ``reduction="none"`` 时，``grad_output`` 既可以是逐样本的上游梯度
        ``(N,)``（与 PyTorch 的 CrossEntropyLoss 一致），也可以是能与
        ``(N, C)`` 广播的数组。
        """
        if self.reduction != "none":
            return super().backward(grad_output)
        if self._grad is None:
            raise RuntimeError("backward() 必须在 forward() 之后调用")
        if grad_output is None:
            return self._grad.copy()

        upstream = np.asarray(grad_output, dtype=np.float64)
        if upstream.ndim == 1 and upstream.size == self._n and self._n != 1:
            upstream = upstream.reshape(-1, 1)
        try:
            grad = self._grad * upstream
        except ValueError:
            raise ValueError(
                f"grad_output 形状 {upstream.shape} 无法与损失形状广播"
            ) from None
        return grad.copy()

    @staticmethod
    def _parse_target(target, n, c, single_sample):
        """把 target 解析为 ``(labels, soft)``，二者恰有一个非 None。"""
        arr = np.asarray(target)

        # 标量标签：只允许单样本
        if arr.ndim == 0:
            if n != 1:
                raise ValueError(f"标量标签只能用于单样本，当前样本数 N={n}")
            return _validate_labels(arr, n, c), None

        # (N, C) 软标签 / one-hot
        if arr.ndim == 2:
            if arr.shape == (n, c):
                return None, _validate_soft_target(arr, n, c)
            if arr.shape == (n, 1):
                return _validate_labels(arr.reshape(-1), n, c), None
            raise ValueError(
                f"target 形状应为 ({n},) 或 ({n}, {c})，收到 {arr.shape}"
            )

        if arr.ndim == 1:
            # 单样本且输入是 (C,) 时，长度为 C 的向量视为软标签
            if single_sample and arr.size == c:
                return None, _validate_soft_target(arr.reshape(1, c), n, c)
            if arr.size == n:
                return _validate_labels(arr, n, c), None
            raise ValueError(
                f"标签数量应为 {n}（或使用长度为 {c} 的软标签），收到 {arr.size}"
            )

        raise ValueError(f"不支持的 target 形状：{arr.shape}")


def build_loss(name, **kwargs):
    """根据配置名创建损失函数，便于 ``config.py`` / 训练入口使用。

    支持 ``"mse"`` / ``"mean_squared_error"`` 与
    ``"cross_entropy"`` / ``"crossentropy"`` / ``"ce"``。

    示例
    ----
    >>> build_loss("cross_entropy", reduction="sum")
    CrossEntropyLoss(reduction='sum')
    """
    key = str(name).strip().lower().replace("-", "_")
    if key in ("mse", "mean_squared_error", "l2"):
        return MSELoss(**kwargs)
    if key in ("cross_entropy", "crossentropy", "ce"):
        return CrossEntropyLoss(**kwargs)
    raise ValueError(f"未知的损失函数：{name!r}（可选：mse、cross_entropy）")
