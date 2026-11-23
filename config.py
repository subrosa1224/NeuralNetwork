"""MNIST 实验配置；命令行参数覆盖 JSON 文件中的同名设置。"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentConfig:
    model: str = "mlp"
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    num_filters: int = 4
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    loss: str = "cross_entropy"
    learning_rate: float = 0.05
    epochs: int = 5
    batch_size: int = 64
    seed: int = 42
    validation_size: int = 5000
    train_limit: Optional[int] = None
    test_limit: Optional[int] = None
    data_dir: str = "data"
    output_dir: Optional[str] = None

    def __post_init__(self):
        if self.model not in ("mlp", "cnn"):
            raise ValueError("model 必须是 mlp 或 cnn")
        if self.loss not in ("cross_entropy", "mse"):
            raise ValueError("loss 必须是 cross_entropy 或 mse")
        for name in ("epochs", "batch_size", "validation_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        for name in ("train_limit", "test_limit"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} 必须是正整数或 null")

    @property
    def run_dir(self):
        return Path(self.output_dir) if self.output_dir is not None else Path("results") / self.model

    def to_dict(self):
        return asdict(self)


def load_config(path=None, **overrides):
    values = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    values.update({key: value for key, value in overrides.items() if value is not None})
    return ExperimentConfig(**values)
