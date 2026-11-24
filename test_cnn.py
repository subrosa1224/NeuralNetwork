import unittest

import numpy as np

from core.losses import CrossEntropyLoss
from models.cnn import CNN


class TestCNN(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)
        self.model = CNN(input_shape=(1, 4, 4), num_filters=2, num_classes=3)
        self.images = np.random.randn(2, 1, 4, 4)
        self.labels = np.array([0, 2])

    def test_forward_and_backward(self):
        logits = self.model.forward(self.images)
        self.assertEqual(logits.shape, (2, 3))
        loss = CrossEntropyLoss()
        loss.forward(logits, self.labels)
        gradient = self.model.backward(loss.backward())
        self.assertEqual(gradient.shape, self.images.shape)
        for parameter, parameter_gradient in self.model.parameters():
            self.assertEqual(parameter.shape, parameter_gradient.shape)

    def test_full_model_gradients(self):
        loss = CrossEntropyLoss()
        loss.forward(self.model.forward(self.images), self.labels)
        input_gradient = self.model.backward(loss.backward()).copy()
        checks = [
            (parameter, gradient.copy(), (0,) * parameter.ndim)
            for parameter, gradient in self.model.parameters()
        ]
        checks.append((self.images, input_gradient, (0, 0, 1, 1)))
        for parameter, gradient, index in checks:
            original = parameter[index]
            parameter[index] = original + 1e-6
            plus = loss.forward(self.model.forward(self.images), self.labels)
            parameter[index] = original - 1e-6
            minus = loss.forward(self.model.forward(self.images), self.labels)
            parameter[index] = original
            np.testing.assert_allclose((plus - minus) / 2e-6, gradient[index], rtol=1e-5, atol=1e-6)

    def test_rejects_missing_channel_dimension(self):
        with self.assertRaises(ValueError):
            self.model.forward(self.images[:, 0])


if __name__ == '__main__':
    unittest.main()
