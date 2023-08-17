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
        op = inv[head]
        assert op is not None
        elements.append(op)

    return elements


def dot4(op1: npt.NDArray, op2: npt.NDArray) -> npt.NDArray:
    """Dot product between rank 4 objects.

    The product is performed considering them as matrices indexed by pairs, so
    linearizing the indices in pairs.

    """
    return np.einsum("aibj,bjck->aick", op1, op2)


def dotop(op1: Operator, op2: Operator) -> Operator:
    r"""Dot product between two operators.

    Essentially a wrapper of :func:`dot4`, applying linear error propagation,
    if applicable.

    Note
    ----
    Linear error propagation requires matrices to be positive before taking the product.

    Indeed, for a simple product of two variables :math:`a \cdot b`, the error is
    propagated in the following way:

    .. math::

        \max_{\sgn_{da}, \sgn{db}} (a + da)(b + db) - ab =
        \max_{\sgn_{da}, \sgn{db}} da \cdot b + a \cdot db + \mathcal{O}(d^2) =
        |da \cdot b| + |a \cdot db| + \mathcal{O}(d^2) =
        |da | \cdot |b| + |a| \cdot |db| + \mathcal{O}(d^2)

    Where the second step is a consequence of being able to vary the two
    variations independently, and last just a trivial property of the product.

    But in a matrix multiplication, each element of the two matrices has an
    independent variation associated. Thus:

    .. math::

        \max_{\sgn_{da_i}, \sgn{db_i}} (a_i + da_i)(b_i + db_i) - a_i b_i =
        \max_{\sgn_{da_i}, \sgn{db_i}} da_i \cdot b_i + a_i \cdot db_i + \mathcal{O}(d^2) =
        |da_i| \cdot |b_i| + |a_i| \cdot |db_i| + \mathcal{O}(d^2)

    """
    val = dot4(op1.operator, op2.operator)

    if op1.error is not None and op2.error is not None:
        err = dot4(np.abs(op1.operator), np.abs(op2.error)) + dot4(
            np.abs(op1.error), np.abs(op2.operator)
        )
    else:
        err = None

    return Operator(val, err)


def join(elements: List[Operator]) -> Operator:
    """Join the elements into the final operator.

    Note
    ----
    Since the matrices composing the path have to be multiplied from the
    destination to the origin, the input order, coming from path (which is
    instead ``origin -> target``), is being reversed.

    .. todo::

        consider if reversing the path...

    """
    return reduce(dotop, reversed(elements))
