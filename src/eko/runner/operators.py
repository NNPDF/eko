"""Combine parts into operators."""
from typing import List

import numpy as np

from ..io.inventory import Inventory
from ..io.items import Operator, Recipe


def retrieve(
    elements: List[Recipe], parts: Inventory, parts_matching: Inventory
) -> List[Operator]:
    """Retrieve parts to be joined."""
    return []


def join(elements: List[Operator]) -> Operator:
    """Join the elements into the final operator."""
    return Operator(np.array([]))
