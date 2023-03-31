"""Operator components."""
from abc import ABC
from dataclasses import dataclass

import numpy.typing as npt

from ..io.dictlike import DictLike
from ..io.types import SquaredScale


@dataclass
class Part(DictLike, ABC):
    """An atomic operator ingredient.

    The operator is always in flavor basis, and on the x-grid specified for
    computation, i.e. it is a rank-4 tensor of shape::

        (flavor basis) x (x-grid) x (flavor basis) x (x-grid)

    """

    operator: npt.NDArray


@dataclass
class Evolution(Part):
    """Evolution in a fixed number of flavors."""

    final: bool
    mu20: SquaredScale
    mu2: SquaredScale


@dataclass
class Matching(Part):
    """Matching conditions between two different flavor schemes, at a given scale."""

    mu2: SquaredScale
