"""Training package exports; fit/evaluate loaded on first use."""
from .optimizer import Optimizer, SGD, Adam, build_optimizer


def __getattr__(name):
    if name in ("fit", "evaluate"):
        from . import trainer
        return getattr(trainer, name)
    raise AttributeError(name)
