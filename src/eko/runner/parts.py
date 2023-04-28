"""Compute operator components."""
import numpy as np

from ..io.items import Operator, Recipe


def compute(recipe: Recipe) -> Operator:
    """Compute EKO component in isolation."""
    return Operator(np.array([]))
