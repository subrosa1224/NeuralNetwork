import numpy as np


class Sigmoid:
    def forward(self, x):
        x = np.asarray(x, dtype=float)
        x = np.clip(x, -500, 500)
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, grad):
        return grad * self.output * (1.0 - self.output)


class ReLU:
    def forward(self, x):
        x = np.asarray(x, dtype=float)
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.input > 0)


class Tanh:
    def forward(self, x):
        x = np.asarray(x, dtype=float)
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad):
        return grad * (1.0 - self.output ** 2)