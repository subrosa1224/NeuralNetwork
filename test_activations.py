import numpy as np
from core.activations import Sigmoid, ReLU, Tanh

x = np.array([-2.0, 0.0, 2.0])
grad = np.ones_like(x)

for name, activation in [
    ("Sigmoid", Sigmoid()),
    ("ReLU", ReLU()),
    ("Tanh", Tanh())
]:
    output = activation.forward(x)
    backward = activation.backward(grad)

    print(f"{name} output: {output}")
    print(f"{name} backward: {backward}")
    print()