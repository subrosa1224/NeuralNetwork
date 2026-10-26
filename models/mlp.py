import numpy as np
from core.layers import Linear
from core.activations import ReLU, Sigmoid, Tanh


class MLP:
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activation="relu",
        output_activation=None
    ):
        activation_map = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh
        }

        activation = activation.lower()

        if activation not in activation_map:
            raise ValueError("Unsupported activation function")

        if output_activation is not None:
            output_activation = output_activation.lower()
            if output_activation not in activation_map:
                raise ValueError("Unsupported output activation")

        sizes = [input_size] + list(hidden_sizes) + [output_size]

        if any(size <= 0 for size in sizes):
            raise ValueError("All layer sizes must be positive")

        self.layers = []
        self.activations = []

        for i in range(len(sizes) - 1):
            self.layers.append(
                Linear(sizes[i], sizes[i + 1])
            )

            if i < len(sizes) - 2:
                self.activations.append(
                    activation_map[activation]()
                )
            elif output_activation is not None:
                self.activations.append(
                    activation_map[output_activation]()
                )
            else:
                self.activations.append(None)

    def forward(self, x):
        current = np.asarray(x, dtype=float)

        self.inputs = []
        self.activations_inputs = []

        for layer, activation in zip(
            self.layers, self.activations
        ):
            self.inputs.append(current)

            current = layer.forward(current)

            self.activations_inputs.append(current)

            if activation is not None:
                current = activation.forward(current)

        return current

    def backward(self, grad_output):
        grad = grad_output

        for i in reversed(range(len(self.layers))):
            activation = self.activations[i]

            if activation is not None:
                grad = activation.backward(grad)

            grad = self.layers[i].backward(grad)

        return grad

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params