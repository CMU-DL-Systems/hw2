"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.if_bias = bias
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, nonlinearity="relu", 
                                           device=device, dtype=dtype, requires_grad=True))
        if self.if_bias:
            self.bias = init.kaiming_uniform(out_features, 1, nonlinearity="relu", 
                                         device=device, dtype=dtype, requires_grad=True)
            self.bias = Parameter(ops.reshape(self.bias, (1, out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = ops.matmul(X, self.weight)
        if self.if_bias:
            res = ops.add(res, ops.broadcast_to(self.bias, res.shape))
        return res
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        X_shape = X.shape
        batches = X_shape[0]
        i = 1
        dim = 1
        while i < len(X_shape):
            dim *= X_shape[i]
            i += 1
        return ops.reshape(X, (batches, dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_one_hot = init.one_hot(logits.shape[-1], y, device=logits.device, dtype=logits.dtype)
        log_sum_exp = ops.logsumexp(logits, axes=(-1,))
        losses = log_sum_exp - (ops.summation(logits * y_one_hot, axes=(-1,)))
        '''Notice: When the tensor is scalar, its dtype will change to float64 after compuation. So, tensor is used here'''
        return ops.summation(losses) / Tensor(logits.shape[0], device=logits.device, dtype=logits.dtype, requires_grad=False)
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            ex = ops.summation(x, axes=(0,)) / x.shape[0]
            ex_broadcast = ops.broadcast_to(ops.reshape(ex, (1, x.shape[1])), x.shape)
            x_minus_ex = x - ex_broadcast
            var = ops.summation(x_minus_ex ** 2, axes=(0,)) / x.shape[0]
            var_broadcast = ops.broadcast_to(ops.reshape(var, (1, x.shape[1])), x.shape)

            self.running_mean = (1 - self.momentum) * self.running_mean.data + self.momentum * ex.data
            self.running_var = (1 - self.momentum) * self.running_var.data + self.momentum * var.data

            x = x_minus_ex / ((var_broadcast + self.eps) ** 0.5)
            return ops.broadcast_to(self.weight, x.shape) * x + ops.broadcast_to(self.bias, x.shape)
        else:
            running_mean_broadcast = ops.broadcast_to(ops.reshape(self.running_mean, (1, x.shape[1])), x.shape)
            running_var_broadcast = ops.broadcast_to(ops.reshape(self.running_var, (1, x.shape[1])), x.shape)
            x = (x - running_mean_broadcast) / ((running_var_broadcast + self.eps) ** 0.5)
            return ops.broadcast_to(self.weight, x.shape) * x + ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        e_x = ops.summation(x, axes=(-1,)) / self.dim
        e_x = ops.broadcast_to(ops.reshape(e_x, (x.shape[0], 1)), x.shape)
        x_minus_ex = x - e_x
        var_x = ops.summation(x_minus_ex ** 2, axes=(-1,)) / self.dim
        std_x = (var_x + self.eps) ** 0.5
        std_x = ops.broadcast_to(ops.reshape(std_x, (x.shape[0], 1)), x.shape)

        x = x_minus_ex / std_x

        return ops.broadcast_to(self.weight, x.shape) * x + ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            bernoulli = np.random.binomial(1, self.p, x.shape)
            bernoulli = (np.negative(bernoulli) + 1) / (1 - self.p)
            return x * bernoulli
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
