from abc import ABC

import numpy as np
from numbers import Number
from typing import Union


class Node(ABC):
    """Base class for all computational nodes in the graph."""

    def __init__(self, inputs: list["Node"], name: str = "Node"):
        self.name = name
        self._inputs = inputs
        self._forward_cache = None
        self._cache_valid = False

    @property
    def inputs(self):
        return self._inputs

    def forward(self):
        """Computes and caches the forward pass result."""

        if self._cache_valid and self._are_input_caches_valid():
            return self._forward_cache

        self._forward_cache = self._forward()
        self._cache_valid = True
        return self._forward_cache

    def _forward(self):
        """Actual logic for computing forward pass result (to be implemented by subclasses)."""

        raise NotImplementedError()

    def backward(self, wrt_node: "Node", output_grad: Union[Number, np.ndarray]):
        """Computes gradients with respect to the inputs (to be implemented by subclasses)."""

        raise NotImplementedError()

    def _are_input_caches_valid(self):
        """Validates cached forward pass results for the subgraph of the current node."""

        for input in self.inputs:
            if (not input._cache_valid) or (not input._are_input_caches_valid()):
                return False
        return True

    def invalidate_cache(self):
        self._cache_valid = False
        self._forward_cache = None

    def __call__(self, *args, **kwds):
        return self.forward()


class UnaryOperator(Node):
    """Base class for all computational nodes with a single input."""

    def __init__(self, input_node: Node, name: str = "BinaryOperator"):
        super().__init__(inputs=[input_node], name=name)

    @property
    def input_node(self):
        return self._inputs[0]

    @input_node.setter
    def input_node(self, node: Node):
        self._inputs[0] = node


class BinaryOperator(Node):
    """Base class for all computational nodes with two inputs."""

    def __init__(self, left_node: Node, right_node: Node, name: str = "BinaryOperator"):
        super().__init__(inputs=[left_node, right_node], name=name)

    @property
    def left_node(self):
        return self._inputs[0]

    @left_node.setter
    def left_node(self, node: Node):
        self._inputs[0] = node

    @property
    def right_node(self):
        return self._inputs[1]

    @right_node.setter
    def right_node(self, node: Node):
        self._inputs[1] = node
