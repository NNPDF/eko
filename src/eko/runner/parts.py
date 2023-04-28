"""Compute operator components."""
import numpy as np

from ..io.items import Operator


def compute(recipe) -> Operator:
    """Compute EKO component in isolation."""
    return Operator(np.array([]))
