import numpy as np
from typing import List, Union
from numpy.typing import NDArray

# Type alias za pojednostavljenje zapisa
Numeric = Union[float, int, np.number]
NumericArray = Union[List[Numeric], NDArray[np.number]]


class Node:
    def forward(self, input: Numeric) -> Numeric:
        # računa izlaz na temelju ulaza
        pass

    def backward(self, output_grad: Numeric = 1) -> Numeric:
        # računa gradijent po ulazu uz dani gradijent po izlazu
        pass


class Power(Node):
    def __init__(self, exponent: Numeric):
        self.exponent = exponent

    def forward(self, input: Numeric) -> Numeric:
        self.input = input
        return np.power(input, self.exponent)

    def backward(self, output_grad: Numeric = 1) -> Numeric:
        return output_grad * self.exponent * np.power(self.input, self.exponent - 1)


class Exp(Node):
    def forward(self, input: Numeric) -> Numeric:
        self.input = input
        return np.exp(input)

    def backward(self, output_grad: Numeric = 1) -> Numeric:
        return output_grad * np.exp(self.input)


class Log(Node):
    def forward(self, input: Numeric) -> Numeric:
        assert input > 0
        self.input = input
        return np.log(self.input)

    def backward(self, output_grad: Numeric = 1) -> Numeric:
        return output_grad * (1 / self.input)


class AddConst(Node):
    def __init__(self, const: Numeric):
        self.const = const

    def forward(self, input: Numeric) -> Numeric:
        self.input = input
        return input + self.const

    def backward(self, output_grad: Numeric = 1) -> Numeric:
        return output_grad


class MultiplyConst(Node):
    def __init__(self, const: Numeric):
        self.const = const

    def forward(self, input: Numeric) -> Numeric:
        self.input = input
        return input * self.const

    def backward(self, output_grad: Numeric = 1) -> Numeric:
        return output_grad * self.const


class PolynomialSum(Node):
    def __init__(self, coeffs: NumericArray):
        # coeffs[i] je koeficijent uz x^i
        self.coeffs = np.array(coeffs)

    def forward(self, input: Numeric) -> Numeric:
        self.input = input
        return np.sum(self.coeffs * input ** np.arange(len(self.coeffs)))

    def backward(self, output_grad: Numeric = 1) -> Numeric:
        """
        suma od i = 1 dok i <= n
        c_i * i * (x ** (i - 1))
        """

        new_coeffs = np.array(self.coeffs[1:]) * np.arange(1, len(self.coeffs))
        exp_input = self.input ** np.arange(len(self.coeffs) - 1)
        return output_grad * np.sum(new_coeffs * exp_input)


class FunctionChain:
    def __init__(self, functions: List[Node]):
        self.functions = functions

    def forward(self, input: Numeric) -> Numeric:
        result = input
        for function in self.functions:
            result = function.forward(result)
        return result

    def backward(self, output_grad: Numeric = 1) -> Numeric:
        grad = output_grad
        for function in self.functions:
            grad = function.backward(grad)
        return grad


def gradient_descent(
    function_chain: FunctionChain,
    initial_x: Numeric,
    param_niter: Numeric = 1e5,
    param_delta: Numeric = 1e-2,
) -> Numeric:
    x = initial_x
    for _ in range(int(param_niter)):
        function_chain.forward(x)
        gradient = function_chain.backward()

        x += -param_delta * gradient
    return x


if __name__ == "__main__":
    f1 = FunctionChain([AddConst(-1), Power(2), Exp(), AddConst(2), Log(), Log()])
    f2 = FunctionChain([Power(2), AddConst(-1), Power(3)])
    f3 = FunctionChain([Power(2), Exp(), AddConst(1), Log()])

    f1_min = gradient_descent(f1, initial_x=9)
    print("f1 ima min. u točki x =", np.round(f1_min, 4))

    f2_min = gradient_descent(f2, initial_x=-0.8)
    print("f2 ima min. u točki x =", np.round(f2_min, 4))

    f3_min = gradient_descent(f3, initial_x=4)
    print("f3 ima min. u točki x =", np.round(f3_min, 4))
