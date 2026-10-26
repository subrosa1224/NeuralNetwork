import numpy as np


class Linear:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))

        self.input = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        self.dW = np.dot(self.input.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input

    def parameters(self):
        return [
            (self.W, self.dW),
            (self.b, self.db)
        ]