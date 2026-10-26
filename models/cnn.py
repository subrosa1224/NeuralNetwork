
import numpy as np

from core.conv import Conv2D
from core.activations import ReLU
from core.layers import Linear


class CNN:
    def __init__(self, input_shape=(1, 28, 28),
                 num_filters=8, kernel_size=3,
                 stride=1, padding=1, num_classes=10):

        in_channels, height, width = input_shape

        self.input_shape = tuple(input_shape)

        # 卷积层
        self.conv = Conv2D(
            in_channels,
            num_filters,
            kernel_size,
            stride,
            padding
        )

        # 调用成员1实现的 ReLU
        self.relu = ReLU()

        # 计算卷积后的特征图尺寸
        conv_height = (
            height + 2*padding - kernel_size
        ) // stride + 1

        conv_width = (
            width + 2*padding - kernel_size
        ) // stride + 1

        if conv_height <= 0 or conv_width <= 0:
            raise ValueError("卷积后的尺寸必须大于0")

        # 调用成员1实现的 Linear
        self.fc = Linear(
            num_filters * conv_height * conv_width,
            num_classes
        )

        self.feature_shape = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)

        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError("CNN 输入形状不正确")

        # 卷积
        x = self.conv.forward(x)

        # ReLU 激活
        x = self.relu.forward(x)

        # 保存形状，供反向传播使用
        self.feature_shape = x.shape

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # 全连接层
        x = self.fc.forward(x)

        return x

    def backward(self, grad_output):
        if self.feature_shape is None:
            raise RuntimeError("请先执行 forward()")

        # 全连接层反向传播
        grad = self.fc.backward(grad_output)

        # Flatten 反向传播
        grad = grad.reshape(self.feature_shape)

        # ReLU 反向传播
        grad = self.relu.backward(grad)

        # 卷积层反向传播
        grad = self.conv.backward(grad)

        return grad

    def parameters(self):
        return (
            self.conv.parameters()
            + self.fc.parameters()
        )
