"""NumPy optimizers for models exposing ``parameters() -> [(value, grad), ...]``."""
import numpy as np


class Optimizer:
    def __init__(self, model, lr=0.01, weight_decay=0.0):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError('lr must be finite and positive')
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
        if not hasattr(model, 'parameters'):
            raise TypeError('model must implement parameters()')
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.steps = 0

    def _parameters(self):
        params = self.model.parameters()
        if not isinstance(params, (list, tuple)):
            params = list(params)
        seen = set()
        result = []
        for parameter, gradient in params:
            if not isinstance(parameter, np.ndarray) or not np.issubdtype(parameter.dtype, np.floating):
                raise TypeError('parameters must be floating-point NumPy arrays')
            if id(parameter) in seen:
                raise ValueError('duplicate parameter array')
            seen.add(id(parameter))
            if gradient is None:
                raise RuntimeError('call model.backward() before optimizer.step()')
            gradient = np.asarray(gradient)
            if gradient.shape != parameter.shape:
                raise ValueError('gradient shape does not match parameter shape')
            if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(parameter)):
                raise FloatingPointError('non-finite parameter or gradient')
            result.append((parameter, gradient.astype(parameter.dtype, copy=False)))
        if not result:
            raise ValueError('model has no trainable parameters')
        return result

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with optional classic momentum, dampening-free and coupled L2 decay."""
    def __init__(self, model, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(model, lr, weight_decay)
        if not np.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError('momentum must be in [0, 1)')
        self.momentum = float(momentum)
        self._velocity = {}

    def step(self):
        params = self._parameters()
        for p, g in params:
            grad = g + self.weight_decay * p if self.weight_decay else g
            if self.momentum:
                key = id(p)
                v = self._velocity.get(key)
                v = np.array(grad, copy=True) if v is None else self.momentum * v + grad
                self._velocity[key] = v
                grad = v
            p -= self.lr * grad
        self.steps += 1


class Adam(Optimizer):
    """Adam with bias correction and coupled L2 weight decay."""
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        super().__init__(model, lr, weight_decay)
        if not (np.isfinite(beta1) and np.isfinite(beta2) and 0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError('beta1 and beta2 must be in [0, 1)')
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be finite and positive')
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self._moments = {}

    def step(self):
        params = self._parameters()
        t = self.steps + 1
        for p, g in params:
            grad = g + self.weight_decay * p if self.weight_decay else g
            key = id(p)
            m, v = self._moments.get(key, (np.zeros_like(p), np.zeros_like(p)))
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            self._moments[key] = (m, v)
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.steps = t


def build_optimizer(name, model, **kwargs):
    key = name.strip().lower().replace('-', '_')
    if key == 'sgd':
        return SGD(model, **kwargs)
    if key in ('momentum', 'sgd_momentum'):
        kwargs.setdefault('momentum', 0.9)
        return SGD(model, **kwargs)
    if key == 'adam':
        return Adam(model, **kwargs)
    raise ValueError(f'unknown optimizer: {name!r}')
