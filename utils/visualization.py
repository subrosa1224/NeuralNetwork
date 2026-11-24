"""保存训练/验证 Loss 和 Accuracy 曲线。"""

from pathlib import Path


def plot_history(history, output_path, show=False):
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        axis.plot(epochs, [row[f"train_{metric}"] for row in history], "o-", label="Train")
        axis.plot(epochs, [row[f"val_{metric}"] for row in history], "o-", label="Validation")
        axis.set(xlabel="Epoch", ylabel=title, title=title, xticks=epochs)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1].set_ylim(0, 1.02)
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)
    return figure
