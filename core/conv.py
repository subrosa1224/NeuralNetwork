import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0):
        if not all(isinstance(v, int) and v > 0 for v in
                   (in_channels, out_channels, kernel_size, stride)):
            raise ValueError("通道数、卷积核大小和步长必须为正整数")
        if not isinstance(padding, int) or padding < 0:
            raise ValueError("padding 必须为非负整数")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He 初始化
        self.W = np.random.randn(
            out_channels, in_channels,
            kernel_size, kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size ** 2))

        self.b = np.zeros(out_channels)

        self.dW = None
        self.db = None

        self.input_shape = None
        self.input_cols = None
        self.output_height = None
        self.output_width = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)

        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("输入维度必须是 (N, C, H, W)")

        n, c, h, w = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        out_h = (h + 2 * p - k) // s + 1
        out_w = (w + 2 * p - k) // s + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError("卷积核尺寸超过输入尺寸")

        # 对输入进行零填充
        x_pad = np.pad(
            x, ((0, 0), (0, 0), (p, p), (p, p))
        )

        # 手动展开卷积窗口（im2col）
        cols = np.empty(
            (n, out_h, out_w, c, k, k),
            dtype=float
        )

        for i in range(k):
            for j in range(k):
                cols[..., i, j] = x_pad[
                    :, :, i:i+s*out_h:s, j:j+s*out_w:s
                ].transpose(0, 2, 3, 1)

        # 保存反向传播所需的信息
        self.input_shape = x.shape
        self.output_height = out_h
        self.output_width = out_w
        self.input_cols = cols.reshape(-1, c * k * k)

        # 将卷积核展开为二维矩阵
        kernel_matrix = self.W.reshape(
            self.out_channels, -1
        )

        # 卷积计算
        out_cols = self.input_cols @ kernel_matrix.T + self.b

        # 恢复输出维度
        out = out_cols.reshape(
            n, out_h, out_w, self.out_channels
        ).transpose(0, 3, 1, 2)

        return out

    def backward(self, grad_output):
        if self.input_cols is None:
            raise RuntimeError("请先执行 forward()")

        n, c, h, w = self.input_shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        out_h = self.output_height
        out_w = self.output_width

        grad_output = np.asarray(grad_output, dtype=float)

        expected_shape = (
            n, self.out_channels, out_h, out_w
        )

        if grad_output.shape != expected_shape:
            raise ValueError(
                f"梯度形状应为 {expected_shape}"
            )

        # 展开输出梯度
        grad_cols = grad_output.transpose(
            0, 2, 3, 1
        ).reshape(-1, self.out_channels)

        # 计算卷积核梯度
        self.dW = (
            grad_cols.T @ self.input_cols
        ).reshape(self.W.shape)

        # 计算偏置梯度
        self.db = np.sum(grad_cols, axis=0)

        # 计算各卷积窗口对应的输入梯度
        grad_input_cols = (
            grad_cols @ self.W.reshape(self.out_channels, -1)
        ).reshape(n, out_h, out_w, c, k, k)

        grad_input_pad = np.zeros(
            (n, c, h + 2*p, w + 2*p),
            dtype=float
        )

        # 将各窗口梯度累加回输入位置
        for i in range(k):
            for j in range(k):
                grad_input_pad[
                    :, :, i:i+s*out_h:s, j:j+s*out_w:s
                ] += grad_input_cols[..., i, j].transpose(
                    0, 3, 1, 2
                )

        # 去掉 padding
        if p == 0:
            return grad_input_pad

        return grad_input_pad[:, :, p:-p, p:-p]

    def parameters(self):
        return [
            (self.W, self.dW),
            (self.b, self.db)
        ]
