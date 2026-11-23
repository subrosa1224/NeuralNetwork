"""Classical mistake-driven binary and multiclass perceptron (NumPy only)."""
import numpy as np


class Perceptron:
    """Multiclass perceptron: for a wrong prediction, update both classes."""

    def __init__(self, learning_rate=1.0, epochs=30, shuffle=True, seed=42, averaged=False):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        self.learning_rate = float(learning_rate)
        self.epochs, self.shuffle, self.seed = epochs, bool(shuffle), seed
        self.averaged = bool(averaged)

    @staticmethod
    def _validate_x(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or 0 in x.shape or not np.all(np.isfinite(x)):
            raise ValueError("features must be a finite, nonempty (N,D) array")
        return x

    def fit(self, x, y):
        x, y = self._validate_x(x), np.asarray(y)
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("labels must be shape (N,)")
        self.classes_, label_ids = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("at least two classes required")
        k, d = len(self.classes_), x.shape[1]
        self.weights_ = np.zeros((k, d))
        self.bias_ = np.zeros(k)
        self.mistakes_ = []
        rng = np.random.default_rng(self.seed)
        w_sum, b_sum, count = np.zeros_like(self.weights_), np.zeros_like(self.bias_), 0
        for _ in range(self.epochs):
            mistakes = 0
            order = rng.permutation(len(x)) if self.shuffle else np.arange(len(x))
            for i in order:
                target = label_ids[i]
                guess = int(np.argmax(self.weights_ @ x[i] + self.bias_))
                if guess != target:
                    delta = self.learning_rate * x[i]
                    self.weights_[target] += delta
                    self.weights_[guess] -= delta
                    self.bias_[target] += self.learning_rate
                    self.bias_[guess] -= self.learning_rate
                    mistakes += 1
                if self.averaged:
                    w_sum += self.weights_
                    b_sum += self.bias_
                    count += 1
            self.mistakes_.append(mistakes)
            if mistakes == 0:
                break
        if self.averaged:
            self.weights_, self.bias_ = w_sum / count, b_sum / count
        return self

    def decision_function(self, x):
        if not hasattr(self, "weights_"):
            raise RuntimeError("call fit first")
        x = self._validate_x(x)
        if x.shape[1] != self.weights_.shape[1]:
            raise ValueError("unexpected feature count")
        return x @ self.weights_.T + self.bias_

    def predict(self, x):
        scores = self.decision_function(x)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, x, y):
        predicted, y = self.predict(x), np.asarray(y)
        if y.shape != predicted.shape:
            raise ValueError("labels must be shape (N,)")
        return float(np.mean(predicted == y))
