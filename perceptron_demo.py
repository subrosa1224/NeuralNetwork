"""Run: python perceptron_demo.py (requires only NumPy)."""
import numpy as np
from models.perceptron import Perceptron


def main():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(200, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    order = rng.permutation(len(y))
    train, test = order[:160], order[160:]
    model = Perceptron(learning_rate=0.1, epochs=100, seed=2026)
    model.fit(x[train], y[train])
    print("Perceptron epochs:", len(model.mistakes_))
    print("Training accuracy:", f"{model.score(x[train], y[train]):.4f}")
    print("Holdout accuracy:", f"{model.score(x[test], y[test]):.4f}")
    print("Final epoch mistakes:", model.mistakes_[-1])


if __name__ == "__main__":
    main()
