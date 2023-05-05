"""Combine parts into operators."""
from functools import reduce
from typing import List

import numpy as np
import numpy.typing as npt

from ..io.inventory import Inventory
from ..io.items import Evolution, Operator, Recipe


def retrieve(
    headers: List[Recipe], parts: Inventory, parts_matching: Inventory
) -> List[Operator]:
    """Retrieve parts to be joined."""
    elements = []
    for head in headers:
        inv = parts if isinstance(head, Evolution) else parts_matching
        elements.append(inv[head])

    return elements


def dot4(op1: npt.NDArray, op2: npt.NDArray) -> npt.NDArray:
    """Dot product between rank 4 objects.

    The product is performed considering them as matrices indexed by pairs, so
    linearizing the indices in pairs.

    """
    return np.einsum("aibj,bjck->aick", op1, op2)


def dotop(op1: Operator, op2: Operator) -> Operator:
    """Dot product between two operators.

    Essentially a wrapper of :func:`dot4`, applying linear error propagation,
    if applicable.

    """
    val = dot4(op1.operator, op2.operator)

    if op1.error is not None and op2.error is not None:
        err = dot4(op1.operator, op2.error) + dot4(op1.error, op2.operator)
    else:
        err = None

    return Operator(val, err)


def join(elements: List[Operator]) -> Operator:
    """Join the elements into the final operator."""
    return reduce(dotop, reversed(elements))
