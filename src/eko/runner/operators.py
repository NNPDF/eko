"""Combine parts into operators."""
from typing import List

from .. import EKO
from ..io.items import Operator, Recipe


def retrieve(eko: EKO, elements: List[Recipe]) -> List[Operator]:
    """Retrieve parts to be joined."""
    return []


def join(eko: EKO, elements: List[Operator]) -> Operator:
    """Join the elements into the final operator."""
    return Operator(None)
