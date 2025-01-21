import numpy as np
from auto_diff import *
from computational_nodes import *
from abc import ABC, abstractmethod
import data


class Model(ABC):
    def __init__(self):
        self._trained = False
        self._loss = None

    @abstractmethod
    def train(
            self,
            X: np.ndarray,
            Y_: np.ndarray,
            loss_function: Node,
            verbose: bool = False,
    ) -> None:
        pass

    def _train(
            self,
            x: Variable,  # input node
            y: Variable,  # output node
            X: np.ndarray,  # input data
            Y: np.ndarray,  # output data
            loss_function: Node, # loss function
            param_niter: int = 10_000,
            batch_size: int = None,
            param_delta: float = 0.001,
            verbose: bool = False,
    ) -> None:
        """
        Implements model training using gradient descent algorithm.
        """
        assert not self._trained, "Model was already trained."

        def gradient_descent_step() -> float:
            # Forward pass
            loss = loss_function()

            # Backward pass
            _, param_grads = compute_gradients(loss_function)

            # Update parametara
            for param, grad in param_grads:
                param.value = param.value - param_delta * grad

            return loss

        N = len(X)

        if batch_size is None:
            x.value = X
            y.value = Y


        for i in range(param_niter):
            if batch_size is None:
                loss = gradient_descent_step()
            else:
                idxs = np.random.choice(N, batch_size, replace=False)
                x.value = X[idxs]
                y.value = Y[idxs]
                loss = gradient_descent_step()

            if verbose and i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")

        if verbose:
            print(f"Final loss: {loss}")

        self._trained = True
        self._loss = loss

    @abstractmethod
    def classify(self, X: np.ndarray) -> np.ndarray:
        pass

    @property
    def loss(self):
        assert self._trained, "Model not trained."

        return self._loss


class LogisticRegression(Model, ABC):
    def __init__(
            self,
            W: Parameter,
            b: Parameter,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.W = W
        self.b = b

    def train(
            self,
            X: np.ndarray,
            Y_: np.ndarray,
            loss_function: Node,
            param_niter: int = 10_000,
            batch_size: int = None,
            param_delta: float = 0.001,
            verbose: bool = False,
    ) -> None:
        """
        Implements model training using gradient descent algorithm.
        """
        Y = data.class_to_onehot(Y_)

        x = Variable(X, name="x")
        y = Variable(Y, name="y")

        super()._train(x, y, X, Y,
                       loss_function,
                       param_niter=param_niter,
                       batch_size=batch_size,
                       param_delta=param_delta,
                       verbose=verbose)

        self.W = self.W
        self.b = self.b

    def classify(self, X: np.ndarray) -> np.ndarray:
        x = Variable(X, name="x")
        scores = AffineTransform(x, self.W, self.b)
        probs = Softmax(scores)()

        return probs
