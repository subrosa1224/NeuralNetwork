"""Member 2 tests run with pytest and only NumPy project core."""
import numpy as np
import pytest
from training.optimizer import SGD, Adam, build_optimizer


class Toy:
    def __init__(self, gradient=2.0):
        self.w = np.array([1.0])
        self.g = np.array([gradient])

    def parameters(self):
        return [(self.w, self.g)]


def test_sgd():
    m = Toy()
    SGD(m, lr=0.1).step()
    np.testing.assert_allclose(m.w, [0.8])


def test_momentum_two_steps():
    m = Toy()
    opt = SGD(m, lr=0.1, momentum=0.9)
    opt.step()
    opt.step()
    np.testing.assert_allclose(m.w, [0.42])


def test_adam_bias_correction():
    m = Toy()
    Adam(m, lr=0.1).step()
    np.testing.assert_allclose(m.w, [0.9], atol=1e-8)


def test_gradient_refreshed_each_step():
    m = Toy()
    opt = SGD(m, lr=0.1)
    opt.step()
    m.g = np.array([-2.0])
    opt.step()
    np.testing.assert_allclose(m.w, [1.0])


def test_grad_none_and_bad_shape():
    m = Toy()
    m.g = None
    with pytest.raises(RuntimeError):
        SGD(m).step()
    m.g = np.array([[3.0]])
    with pytest.raises(ValueError):
        SGD(m).step()


def test_factory_and_invalid_hyperparameters():
    m = Toy()
    assert isinstance(build_optimizer("momentum", m), SGD)
    assert isinstance(build_optimizer("adam", m), Adam)
    with pytest.raises(ValueError):
        SGD(m, lr=-1)
    with pytest.raises(ValueError):
        Adam(m, beta2=1)


def test_simple_linear_regression_loss_decreases():
    # Equivalent to the established project interface: parameters() returns (array, grad).
    class LinearToy:
        def __init__(self):
            self.w = np.array([0.0])
            self.g = None
        def parameters(self):
            return [(self.w, self.g)]
    model = LinearToy()
    opt = Adam(model, lr=0.05)
    xs = np.linspace(-1, 1, 20)
    target = 2 * xs
    before = np.mean((xs * model.w[0] - target) ** 2)
    for _ in range(100):
        prediction = xs * model.w[0]
        model.g = np.array([2 * np.mean((prediction - target) * xs)])
        opt.step()
    after = np.mean((xs * model.w[0] - target) ** 2)
    assert after < before * 0.01
