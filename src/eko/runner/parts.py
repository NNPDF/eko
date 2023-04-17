"""Operator components."""
from abc import ABC
from dataclasses import dataclass
from typing import Optional

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
    error: Optional[npt.NDArray]


@dataclass
class Evolution(Part):
    """Evolution in a fixed number of flavors."""

    final: bool
    """Whether the operator is the terminal segment of evolution.

    If it is not final, then the operator is an intermediate one.
    Intermediate ones always have final scales :attr:`mu2` corresponding to
    matching scales, and initial scales :attr:`mu20` corresponding to either
    matching scales or the global initial scale of the EKO.

    """
    mu20: SquaredScale
    """Initial scale."""
    mu2: SquaredScale
    """Final scale."""


@dataclass
class Matching(Part):
    """Matching conditions between two different flavor schemes, at a given scale."""

    mu2: SquaredScale
    """Matching scale."""
