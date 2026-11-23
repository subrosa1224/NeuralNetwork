"""由同一份配置构建模型，并适配成员 5 返回的 MNIST 图片。"""

import numpy as np

from models.cnn import CNN
from models.mlp import MLP


def build_model(config):
    if config.model == "mlp":
        return MLP(784, config.hidden_sizes, 10, activation=config.activation)
    if config.model == "cnn":
        return CNN(
            input_shape=(1, 28, 28), num_filters=config.num_filters,
            kernel_size=config.kernel_size, stride=config.stride,
            padding=config.padding, num_classes=10,
        )
    raise ValueError(f"未知模型：{config.model}")


def prepare_inputs(images, model_name):
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("MNIST 原图形状必须是 (N, 28, 28)")
    if model_name == "mlp":
        return images.reshape(images.shape[0], 784)
    if model_name == "cnn":
        return images[:, np.newaxis, :, :]
    raise ValueError(f"未知模型：{model_name}")
