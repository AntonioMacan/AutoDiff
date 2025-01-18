from base_nodes import Node, UnaryOperator, BinaryOperator
from numbers import Number
import numpy as np
from typing import Union


def standardize_value(value: Union[Number, np.ndarray]):
    """Converts scalar or one-dimensional array inputs into column vectors."""

    if isinstance(value, Number):
        return np.array([[value]])
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return value[:, np.newaxis]
    return value


class Variable(Node):
    """Represents a variable node."""

    def __init__(self, value: Union[Number, np.ndarray], name: str = "Variable"):
        super().__init__(inputs=[], name=name)
        self._value = standardize_value(value)

    @property
    def value(self):
        return self.forward()

    @value.setter
    def value(self, new_value):
        self.invalidate_cache()
        self._value = standardize_value(new_value)

    def _forward(self):
        return self._value

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        grad_input = output_grad if wrt_node == self else 0
        return grad_input, []


class Add(BinaryOperator):
    """Represents the addition operation."""

    def __init__(self, left_node: Node, right_node: Node, name: str = "Add"):
        super().__init__(left_node, right_node, name)

    def _forward(self):
        return self.left_node() + self.right_node()

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node not in (self.left_node, self.right_node):
            grad_input = np.zeros_like(output_grad)
        else:
            grad_input = output_grad

        return grad_input, []


class Sub(BinaryOperator):
    """Represents the subtraction operation."""

    def __init__(self, left_node: Node, right_node: Node, name: str = "Sub"):
        super().__init__(left_node, right_node, name)

    def _forward(self):
        return self.left_node() - self.right_node()

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node == self.left_node:
            grad_input = output_grad
        elif wrt_node == self.right_node:
            grad_input = -output_grad
        else:
            grad_input = np.zeros_like(output_grad)

        return grad_input, []


class Neg(UnaryOperator):
    """Represents the negation operation."""

    def __init__(self, input_node: Node, name: str = "Neg"):
        super().__init__(input_node, name)

    def _forward(self):
        return -self.input_node()

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node != self.input_node:
            grad_input = np.zeros_like(output_grad)
        else:
            grad_input = output_grad * (-1)

        return grad_input, []


class Mul(BinaryOperator):
    """Represents the element-wise multiplication."""

    def __init__(self, left_node: Node, right_node: Node, name: str = "Mul"):
        super().__init__(left_node, right_node, name)

    def _forward(self):
        return self.left_node() * self.right_node()

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node == self.left_node:
            grad_input = output_grad * self.right_node()
        elif wrt_node == self.right_node:
            grad_input = output_grad * self.left_node()
        else:
            grad_input = np.zeros_like(output_grad)

        return grad_input, []


class Div(BinaryOperator):
    """Represents the element-wise division."""

    def __init__(self, left_node: Node, right_node: Node, name: str = "Div"):
        super().__init__(left_node, right_node, name)

    def _forward(self):
        return self.left_node() / self.right_node()

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node == self.left_node:
            grad_input = output_grad / self.right_node()
        elif wrt_node == self.right_node:
            grad_input = output_grad * (-self.left_node() / (self.right_node() ** 2))
        else:
            grad_input = np.zeros_like(output_grad)

        return grad_input, []


class PowConst(UnaryOperator):
    """Represents the power operation with a constant exponent."""

    def __init__(self, input_node: Node, const: Number, name: str = "PowConst"):
        super().__init__(input_node, name)
        self.const = const

    def _forward(self):
        return self.input_node() ** self.const

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node != self.input_node:
            grad_input = np.zeros_like(output_grad)
        else:
            grad_input = (
                output_grad * self.const * self.input_node() ** (self.const - 1)
            )

        return grad_input, []


class Exp(UnaryOperator):
    """Represents the exponential function."""

    def __init__(self, input_node: Node, name: str = "Exp"):
        super().__init__(input_node, name)

    def _forward(self):
        return np.exp(self.input_node())

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node != self.input_node:
            grad_input = np.zeros_like(output_grad)
        else:
            grad_input = output_grad * np.exp(self.input_node())

        return grad_input, []


class Log(UnaryOperator):
    """Represents the natural logarithm function."""

    def __init__(self, input_node: Node, name: str = "Log"):
        super().__init__(input_node, name)

    def _forward(self):
        if any(self.input_node() == 0):
            raise RuntimeError()
        return np.log(self.input_node())

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node != self.input_node:
            grad_input = np.zeros_like(output_grad)
        else:
            grad_input = output_grad * (1 / self.input_node())

        return grad_input, []


class AddConst(UnaryOperator):
    """Represents the operation that adds constant to the input node's value."""

    def __init__(self, input_node: Node, const: Number, name: str = "AddConst"):
        super().__init__(input_node, name)
        self.const = const

    def _forward(self):
        return self.input_node() + self.const

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node != self.input_node:
            grad_input = np.zeros_like(output_grad)
        else:
            grad_input = output_grad

        return grad_input, []


class MulConst(UnaryOperator):
    """Represents the operation that multiplies input node's value by a constant."""

    def __init__(self, input_node: Node, const: Number, name: str = "MulConst"):
        super().__init__(input_node, name)
        self.const = const

    def _forward(self):
        return self.input_node() * self.const

    def backward(
        self, wrt_node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node != self.input_node:
            grad_input = np.zeros_like(output_grad)
        else:
            grad_input = output_grad * self.const

        return grad_input, []


class PolynomialSum(UnaryOperator):
    """
    Represents a polynomial sum operation on the input node's value.

    Given an input value \( x \) and a set of constants \( c_i \), the operation computes:

    c_0 + c_1 * x^1 + c_2 * x^2 + ... + c_n * x^n

    Where:
    - \( x \): The value from the input node.
    - \( c_i \): Constants defining the polynomial coefficients.
    """

    def __init__(
        self, input_node: Node, consts: np.ndarray, name: str = "PolynomialSum"
    ):
        super().__init__(input_node, name)
        self.consts = consts

    def _forward(self):
        input_value = self.input_node()

        powers = np.power.outer(input_value, np.arange(len(self.consts)))
        result = np.sum(self.consts * powers, axis=-1)

        return result

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node != self.input_node:
            grad_input = np.zeros_like(output_grad)
        else:
            input_value = self.input_node()
            powers = np.power.outer(input_value, np.arange(len(self.consts) - 1))
            new_consts = self.consts[1:] * np.arange(1, len(self.consts))
            polysum = np.sum(new_consts * powers, axis=-1)
            grad_input = output_grad * polysum

        return grad_input, []


class MatMul(BinaryOperator):
    """Represents the matrix multiplication."""

    def __init__(self, left_node: Node, right_node: Node, name="BinaryOperator"):
        super().__init__(left_node, right_node, name)

    def _forward(self):
        return np.matmul(self.left_node(), self.right_node())

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node == self.left_node:
            grad_input = np.matmul(output_grad, self.right_node().T)
        elif wrt_node == self.right_node:
            grad_input = np.matmul(self.left_node().T, output_grad)
        else:
            grad_input = np.zeros_like(output_grad)

        return grad_input, []


class AffineTransform(UnaryOperator):
    """Represents the affine transformation."""

    def __init__(
        self,
        input_node: Node,
        weights: np.ndarray,
        bias: np.ndarray,
        name: str = "AffineTransform",
    ):
        super().__init__(input_node, name)
        self.weights = weights
        self.bias = bias

    def _forward(self):
        x = self.input_node()
        return np.matmul(x, self.weights) + self.bias

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node != self.input_node:
            grad_input = 0
        else:
            grad_input = np.matmul(output_grad, self.weights.T)

        x = self.input_node()
        grad_weights = np.matmul(x.T, output_grad)
        grad_bias = np.sum(output_grad, axis=0, keepdims=True)

        grad_params = [(self.weights, grad_weights), (self.bias, grad_bias)]
        return grad_input, grad_params


class Sigmoid(UnaryOperator):
    """Represents the sigmoid function."""

    def __init__(self, input_node, name="Sigmoid"):
        super().__init__(input_node, name)

    def _forward(self):
        x = self.input_node()
        return 1 / (1 + np.exp(-x))

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node == self.input_node:
            sigmoid = self()
            grad_input = output_grad * sigmoid * (1 - sigmoid)
        else:
            grad_input = np.zeros_like(output_grad)

        return grad_input, []


class ReLU(UnaryOperator):
    """Represents the ReLU function."""

    def __init__(self, input_node, name="ReLU"):
        super().__init__(input_node, name)

    def _forward(self):
        return np.maximum(0, self.input_node())

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node == self.input_node:
            grad_input = output_grad * (self.input_node() > 0)
        else:
            grad_input = np.zeros_like(output_grad)

        return grad_input, []


class Softmax(UnaryOperator):
    """Represents the softmax function."""

    def __init__(self, input_node: Node, name: str = "Softmax"):
        super().__init__(input_node, name)

    def _forward(self):
        s = self.input_node()
        stable_values = s - np.max(s, axis=1, keepdims=True)
        exps = np.exp(stable_values)
        exps_sum = np.sum(exps, axis=-1, keepdims=True)
        return exps / exps_sum

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        if wrt_node != self.input_node:
            grad_input = np.zeros_like(output_grad)
        else:
            softmax = self()
            grad_input = softmax * (
                output_grad - np.sum(output_grad * softmax, axis=-1, keepdims=True)
            )

        return grad_input, []


class SoftmaxCrossEntropyWithLogits(BinaryOperator):
    """Represents the softmax cross entropy with logits function."""

    def __init__(
        self, logits_node: Node, labels_node: Node, name="SoftmaxCrossEntropyWithLogits"
    ):
        super().__init__(left_node=logits_node, right_node=labels_node, name=name)

    logits_node = BinaryOperator.left_node
    labels_node = BinaryOperator.right_node

    def _forward(self):
        logits = self.logits_node()
        labels = self.labels_node()

        # Softmax stabilization
        stable_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(stable_logits)
        exps_sum = np.sum(exps, axis=-1, keepdims=True)
        probs = exps / exps_sum

        # Cross-entropy loss
        # A small constant (1e-9) is added to predicted probabilities
        # to prevent numerical instability when computing the logarithm,
        # especially if any probability is zero.
        loss = -np.sum(labels * np.log(probs + 1e-9), axis=-1)
        return np.mean(loss)

    def backward(
        self, wrt_node: Node, output_grad: Union[Number, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output_grad = standardize_value(output_grad)

        logits = self.logits_node()
        labels = self.labels_node()

        stable_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(stable_logits)
        exps_sum = np.sum(exps, axis=-1, keepdims=True)
        probs = exps / exps_sum

        if wrt_node == self.logits_node:
            grad_input = (probs - labels) * output_grad / probs.shape[0]
        elif wrt_node == self.labels_node:
            grad_input = -np.log(probs + 1e-9) * output_grad / probs.shape[0]
        else:
            grad_input = np.zeros_like(output_grad)

        return grad_input, []
