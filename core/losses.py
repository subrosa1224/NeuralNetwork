import numpy as np


def _to_float(x):
    return np.asarray(x, dtype=float)


def _log_softmax(x):
    # 减去最大值，避免 exp 溢出
    shifted = x - np.max(x, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _validate_labels(labels, n, c):
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if labels.size != n:
        raise ValueError(f"标签数量应为 {n}，收到 {labels.size}")
    if not np.all(np.isfinite(labels)):
        raise ValueError("标签中不能包含 inf / nan")
    if np.any(labels != np.floor(labels)):
        raise ValueError("类别下标必须是整数")
    labels = labels.astype(np.int64)
    if np.any(labels < 0) or np.any(labels >= c):
        raise ValueError(f"类别下标必须在 [0, {c - 1}] 范围内")
    return labels


def _validate_soft_target(target, n, c):
    target = np.asarray(target, dtype=float)
    if target.shape != (n, c):
        raise ValueError(f"软标签形状应为 {(n, c)}，收到 {target.shape}")
    if not np.all(np.isfinite(target)):
        raise ValueError("软标签中不能包含 inf / nan")
    if np.any(target < 0):
        raise ValueError("软标签中的概率不能为负数")
    return target


class Loss:
    def __init__(self, reduction="mean"):
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction 必须是 'mean'、'sum' 或 'none'，收到 {reduction!r}"
            )
        self.reduction = reduction
        self.last_loss = None
        self._grad = None
        self._loss_count = 1

    def forward(self, pred, target):
        raise NotImplementedError

    def backward(self, grad_output=None):
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
                    f"reduction={self.reduction!r} 时 grad_output 必须是标量"
                )
            return self._grad * float(scale.reshape(-1)[0])
        return self._grad.copy()

    def __call__(self, pred, target):
        return self.forward(pred, target)

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction={self.reduction!r})"

    def _reduce(self, values):
        self._loss_count = max(int(values.size), 1)
        if self.reduction == "mean":
            return float(np.mean(values))
        if self.reduction == "sum":
            return float(np.sum(values))
        return values

    def _scale_grad(self, grad):
        if self.reduction == "mean":
            return grad / self._loss_count
        return grad

    def _invalidate(self):
        self._grad = None
        self.last_loss = None


class MSELoss(Loss):
    def forward(self, pred, target):
        self._invalidate()
        pred = _to_float(pred)
        target = _to_float(target)
        if pred.shape != target.shape:
            raise ValueError(
                f"MSE 要求 pred 与 target 形状一致，收到 {pred.shape} 和 {target.shape}"
            )

        diff = pred - target
        self.last_loss = self._reduce(diff * diff)
        self._grad = self._scale_grad(2.0 * diff)
        return self.last_loss


class CrossEntropyLoss(Loss):
    # pred 默认是 logits，target 可以是类别下标或 one-hot/软标签
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
            # 硬标签：只取正确类别的对数概率
            values = -log_probs[rows, labels]
            if self.from_logits:
                grad = probs.copy()
                grad[rows, labels] -= 1.0
            else:
                grad = np.zeros_like(probs)
                grad[rows, labels] = -1.0 / probs[rows, labels]
        else:
            # 软标签：按目标概率加权求和
            positive = soft > 0
            target_masked = np.where(positive, soft, 0.0)
            log_masked = np.where(positive, log_probs, 0.0)
            values = -np.sum(target_masked * log_masked, axis=1)

            if self.from_logits:
                # dL/dz = p * sum(t) - t
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
        if self.reduction != "none":
            return super().backward(grad_output)
        if self._grad is None:
            raise RuntimeError("backward() 必须在 forward() 之后调用")
        if grad_output is None:
            return self._grad.copy()

        upstream = np.asarray(grad_output, dtype=float)
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
        arr = np.asarray(target)

        # 标量标签只用于单样本
        if arr.ndim == 0:
            if n != 1:
                raise ValueError(f"标量标签只能用于单样本，当前样本数 N={n}")
            return _validate_labels(arr, n, c), None

        # (N, C) 软标签，或 (N, 1) 的标签列
        if arr.ndim == 2:
            if arr.shape == (n, c):
                return None, _validate_soft_target(arr, n, c)
            if arr.shape == (n, 1):
                return _validate_labels(arr.reshape(-1), n, c), None
            raise ValueError(
                f"target 形状应为 ({n},) 或 ({n}, {c})，收到 {arr.shape}"
            )

        # 单样本且 pred 是 (C,) 时，长度为 C 的向量按软标签处理
        if arr.ndim == 1:
            if single_sample and arr.size == c:
                return None, _validate_soft_target(arr.reshape(1, c), n, c)
            if arr.size == n:
                return _validate_labels(arr, n, c), None
            raise ValueError(
                f"标签数量应为 {n}（或使用长度为 {c} 的软标签），收到 {arr.size}"
            )

        raise ValueError(f"不支持的 target 形状：{arr.shape}")


def build_loss(name, **kwargs):
    key = str(name).strip().lower().replace("-", "_")
    if key in ("mse", "mean_squared_error", "l2"):
        return MSELoss(**kwargs)
    if key in ("cross_entropy", "crossentropy", "ce"):
        return CrossEntropyLoss(**kwargs)
    raise ValueError(f"未知的损失函数：{name!r}（可选：mse、cross_entropy）")
