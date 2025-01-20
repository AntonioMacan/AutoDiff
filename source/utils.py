import numpy as np
from computational_nodes import Node


def topological_sort(node: Node):
    """Returns a topological ordering of the graph nodes."""

    visited = set()
    topo = []

    def visit(n: Node):
        if n not in visited:
            visited.add(n)
            for input in n.inputs:
                visit(input)
            topo.append(n)

    visit(node)
    return topo


def reversed_topological_sort(node: Node):
    """Returns the reverse topological order."""

    return list(reversed(topological_sort(node)))
