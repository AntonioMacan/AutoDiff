import numpy as np
from base_nodes import Node
from computational_nodes import Variable
from numbers import Number
from typing import Union
from utils import reversed_topological_sort


def compute_gradients(output_node: Node, topo_sorted_nodes: list[Node] = None):
    """Computes gradients for each node in the computational graph."""

    if topo_sorted_nodes is None:
        topo_sorted_nodes = reversed_topological_sort(output_node)

    input_grads = {node: np.zeros_like(node()) for node in topo_sorted_nodes}
    input_grads[output_node] = np.ones_like(output_node())  # Start gradient from output

    param_grads = {}

    for node in topo_sorted_nodes:
        for input in node.inputs:
            grad_input, grad_params = node.backward(input, input_grads[node])
            input_grads[input] = input_grads[input] + grad_input

            for param, grad in grad_params:
                # Unique ID parameter is used as key
                # since parameters are of np.ndarray type
                param_id = id(param)
                if param_id not in param_grads:
                    param_grads[param_id] = (param, grad)
                else:
                    prev_grad = param_grads[param_id][1]
                    param_grads[param_id] = (param, prev_grad + grad)

        node.invalidate_cache()

    param_grads_list = list(param_grads.values())
    return input_grads, param_grads_list


def gradient_descent_input(
    output_node: Node,
    variable: Variable,
    initial_x: Union[Number, np.ndarray],
    param_niter: int = int(1e5),
    param_delta: np.float64 = np.float64(1e-2),
) -> Number:
    """Performs gradient descent optimization on an input variable."""

    variable.value = initial_x
    topo_sorted_nodes = reversed_topological_sort(output_node)
    for _ in range(int(param_niter)):
        # Forward pass
        output_node()

        # Gradient computation
        gradients = compute_gradients(output_node, topo_sorted_nodes)[0]
        grad = gradients[variable]

        variable.value = variable.value - param_delta * grad

    return variable.value


def gradient_descent_params(
    output_node: Node,
    param_niter: int = int(1e5),
    param_delta: np.float64 = np.float64(1e-2),
) -> None:
    """Performs gradient descent optimization on parameters of the graph."""

    topo_sorted_nodes = reversed_topological_sort(output_node)

    for _ in range(param_niter):
        output_node()

        _, param_grads = compute_gradients(output_node, topo_sorted_nodes)

        for param, grad in param_grads:
            param -= param_delta * grad

        for node in topo_sorted_nodes:
            node.invalidate_cache()
