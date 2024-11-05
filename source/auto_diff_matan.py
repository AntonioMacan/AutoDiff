import numpy as np


class Node:
    def forward(self, input: float) -> float:
        
        # raˇcuna izlaz na temelju ulaza
        pass

    def backward(self, output_grad: float = 1.0) -> float:
        # raˇcuna gradijent po ulazu uz dani gradijent po izlazu
        pass


class Power(Node):

    def __init__(self, exponent: float) -> None:
        super().__init__()
        self.exponent: float = exponent

    def forward(self, input: float) -> float:
        self.input: float = input

        return np.pow(self.input, self.exponent)

    def backward(self, output_grad: float = 1) -> float:
        return output_grad * self.exponent * np.pow(self.input,
                                                    self.exponent - 1)


class Log(Node):
    def forward(self, input: float) -> float:
        assert input > 0

        self.input: float = input

        return np.log(self.input)

    def backward(self, output_grad: float = 1) -> float:
        return output_grad * (1 / self.input)


class Exp(Node):
    def forward(self, input: float) -> float:
        self.input: float = input

        return np.exp(input)

    def backward(self, output_grad: float = 1) -> float:
        return output_grad * np.exp(self.input)


class PlusConst(Node):
    def __init__(self, constant: float) -> None:
        super().__init__()
        self.constant: float = constant

    def forward(self, input: float) -> float:
        self.input: float = input

        return self.input + self.constant

    def backward(self, output_grad: float = 1) -> float:
        return output_grad


class MultiplyConst(Node):
    def __init__(self, constant: float) -> None:
        super().__init__()
        self.constant: float = constant

    def forward(self, input: float) -> float:
        self.input: float = input

        return self.input * self.constant

    def backward(self, output_grad: float = 1) -> float:
        return output_grad * self.constant


class Polynomial(Node):
    def __init__(self, coefficients: list[float]) -> None:
        super().__init__()
        self.coefficients: np.array = np.array(coefficients)
        self.powers: np.array = np.arange(len(self.coefficients) + 1)

    def forward(self, input: float) -> float:
        self.input: float = input

        return np.sum(self.input * self.coefficients ** self.powers)

    def backward(self, output_grad: float = 1) -> float:
        """
        suma od i = 1 dok i <= n
        cn * i * (x ** (i - 1))
        """
        return output_grad * np.sum(self.coefficients[1:] * self.powers[1:]
                                    * self.input ** self.powers[1:])


class FunctionChain:
    def __init__(self, functions: list[Node]) -> None:
        self.functions = functions

    def forward(self, input: float) -> float:
        for f in self.functions:
            input = f.forward(input)

        return input

    def backward(self, output_grad: float) -> float:
        for f in self.functions:
            output_grad = f.backward(output_grad)

        return output_grad


if __name__ == "__main__":
    f1 = FunctionChain([PlusConst(-1), Power(2), Exp(), PlusConst(2), Log(),
                        Log()])
    f2 = FunctionChain([Power(2), PlusConst(-1), Power(3)])
    f3 = FunctionChain([Power(2), Exp(), PlusConst(1), Log()])
