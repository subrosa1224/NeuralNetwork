
import unittest
import numpy as np
from core.conv import Conv2D


class TestConv2D(unittest.TestCase):

    def test_forward(self):
        conv = Conv2D(1, 1, 2)

        conv.W[:] = np.array([
            [[[1.0, 0.0],
              [0.0, -1.0]]]
        ])
        conv.b[:] = 0

        x = np.arange(1, 10, dtype=float)
        x = x.reshape(1, 1, 3, 3)

        y = conv.forward(x)

        expected = np.full((1, 1, 2, 2), -4.0)

        np.testing.assert_allclose(y, expected)

    def test_multi_channel(self):
        conv = Conv2D(
            2, 3, 3,
            stride=2,
            padding=1
        )

        x = np.random.randn(2, 2, 5, 6)

        y = conv.forward(x)

        self.assertEqual(y.shape, (2, 3, 3, 3))

        dx = conv.backward(np.ones_like(y))

        self.assertEqual(dx.shape, x.shape)

        for param, grad in conv.parameters():
            self.assertEqual(param.shape, grad.shape)

    def test_gradient(self):
        np.random.seed(3)

        conv = Conv2D(2, 3, 3, stride=2, padding=1)
        x = np.random.randn(2, 2, 4, 5)

        y = conv.forward(x)
        upstream = np.random.randn(*y.shape)

        dx = conv.backward(upstream)

        dW = conv.dW.copy()
        db = conv.db.copy()

        eps = 1e-6

        def loss():
            return np.sum(conv.forward(x) * upstream)

        # 检查卷积核梯度
        idx = (0, 0, 0, 0)
        old = conv.W[idx]

        conv.W[idx] = old + eps
        plus = loss()

        conv.W[idx] = old - eps
        minus = loss()

        conv.W[idx] = old

        numerical = (plus - minus) / (2 * eps)

        np.testing.assert_allclose(
            numerical, dW[idx], rtol=1e-5, atol=1e-6
        )

        # 检查偏置梯度
        idx = 0
        old = conv.b[idx]

        conv.b[idx] = old + eps
        plus = loss()

        conv.b[idx] = old - eps
        minus = loss()

        conv.b[idx] = old

        numerical = (plus - minus) / (2 * eps)

        np.testing.assert_allclose(
            numerical, db[idx], rtol=1e-5, atol=1e-6
        )

        # 检查输入梯度
        idx = (0, 0, 0, 0)
        old = x[idx]

        x[idx] = old + eps
        plus = loss()

        x[idx] = old - eps
        minus = loss()

        x[idx] = old

        numerical = (plus - minus) / (2 * eps)

        np.testing.assert_allclose(
            numerical, dx[idx], rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
