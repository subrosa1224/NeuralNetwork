"""统一训练/验证入口；与成员6现有接口保持兼容。"""
import numpy as np
from core.losses import MSELoss
from utils.metrics import calculate_metrics
from .optimizer import SGD, Adam, build_optimizer


def loss_targets(labels, logits, loss_fn):
    """MSE 采用 one-hot 目标，交叉熵采用整数类别。"""
    if isinstance(loss_fn, MSELoss):
        return np.eye(logits.shape[1])[np.asarray(labels, dtype=int)]
    return labels


def _run_epoch(model, loader, loss_fn, optimizer=None):
    total_loss = 0.0
    labels, predictions = [], []
    for images, target in loader:
        logits = model.forward(images)
        batch_loss = loss_fn.forward(logits, loss_targets(target, logits, loss_fn))
        if not np.isfinite(batch_loss):
            raise FloatingPointError(f"损失不是有限值：{batch_loss}")
        if optimizer is not None:
            model.backward(loss_fn.backward())
            optimizer.step()
        total_loss += float(batch_loss) * len(target)
        labels.append(target)
        predictions.append(logits.argmax(axis=1))
    if not labels:
        raise ValueError("训练或评估数据不能为空")
    truth = np.concatenate(labels)
    metrics = calculate_metrics(truth, np.concatenate(predictions), labels=np.arange(10))
    metrics["loss"] = float(total_loss / len(truth))
    return metrics


def evaluate(model, loader, loss_fn):
    return _run_epoch(model, loader, loss_fn)


def fit(model, train_loader, validation_loader, loss_fn, epochs, optimizer):
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    history = []
    for epoch in range(1, epochs + 1):
        train = _run_epoch(model, train_loader, loss_fn, optimizer)
        validation = evaluate(model, validation_loader, loss_fn)
        history.append({
            "epoch": epoch,
            "train_loss": train["loss"], "train_accuracy": train["accuracy"],
            "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
        })
        print(
            f"epoch {epoch}/{epochs}  loss={train['loss']:.4f}  "
            f"accuracy={train['accuracy']:.4f}  val_loss={validation['loss']:.4f}  "
            f"val_accuracy={validation['accuracy']:.4f}", flush=True,
        )
    return history
