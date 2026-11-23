"""Member 2 classical perceptron tests."""
import numpy as np
import pytest
from models.perceptron import Perceptron


def test_binary_linearly_separable():
    x = np.array([[-3., -2.], [-2., -3.], [-2., -1.],
                  [2., 3.], [3., 2.], [1., 2.]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = Perceptron(epochs=30, seed=7).fit(x, y)
    assert model.score(x, y) == 1.0
    assert model.mistakes_[-1] == 0


def test_multiclass_training():
    x = np.array([[4., 0.], [5., 1.], [0., 4.], [1., 5.],
                  [-4., -4.], [-5., -3.]])
    y = np.array([10, 10, 20, 20, 30, 30])
    model = Perceptron(epochs=100, seed=4).fit(x, y)
    assert model.score(x, y) == 1.0
    assert model.decision_function(x).shape == (6, 3)


def test_reproducible():
    x = np.array([[1., 1.], [-1., -1.], [2., -1.], [-1., 2.]])
    y = np.array([1, 0, 1, 0])
    a = Perceptron(epochs=10, seed=5).fit(x, y)
    b = Perceptron(epochs=10, seed=5).fit(x, y)
    np.testing.assert_array_equal(a.weights_, b.weights_)
    assert a.mistakes_ == b.mistakes_


def test_averaged_perceptron():
    x = np.array([[-2., -2.], [2., 2.], [-3., -1.], [3., 1.]])
    y = np.array([0, 1, 0, 1])
    model = Perceptron(epochs=30, averaged=True).fit(x, y)
    assert model.score(x, y) == 1.0


def test_errors():
    with pytest.raises(ValueError):
        Perceptron(learning_rate=0)
    with pytest.raises(ValueError):
        Perceptron(epochs=0)
    with pytest.raises(RuntimeError):
        Perceptron().predict([[0., 1.]])
    with pytest.raises(ValueError):
        Perceptron().fit([[1., 2.]], [1])
    with pytest.raises(ValueError):
        Perceptron().fit([[float('nan'), 2.], [1., 1.]], [0, 1])
