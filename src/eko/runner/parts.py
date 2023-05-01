"""Compute operator components."""
import numpy as np

from .. import EKO
from ..io.items import Operator


def compute(eko: EKO, recipe) -> Operator:
    """Compute EKO component in isolation."""
    return Operator(np.array([]))
