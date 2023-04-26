"""Operator components."""
from abc import ABC
from dataclasses import dataclass

from eko.io.struct import Operator

from ..io.dictlike import DictLike
from ..io.types import SquaredScale


@dataclass
class Part(DictLike, ABC):
    """An atomic operator ingredient.

    The operator is always in flavor basis, and on the x-grid specified for
    computation, i.e. it is a rank-4 tensor of shape::

        (flavor basis) x (x-grid) x (flavor basis) x (x-grid)

    """

    operator: Operator


@dataclass
class Evolution(Part):
    """Evolution in a fixed number of flavors."""

    cliff: bool
    """Whether the operator is reaching a matching scale.

    Cliff operators are the only ones allowed to be intermediate, even though
    they can also be final segments of an evolution path (see
    :meth:`eko.matchings.Atlas.path`).

    Intermediate ones always have final scales :attr:`mu2` corresponding to
    matching scales, and initial scales :attr:`mu20` corresponding to either
    matching scales or the global initial scale of the EKO.

    Note
    ----

    The name of *cliff* operators stems from the following diagram::

        nf = 3 --------------------------------------------------------
                        |
        nf = 4 --------------------------------------------------------
                                |
        nf = 5 --------------------------------------------------------
                                                            |
        nf = 6 --------------------------------------------------------

    where each lane corresponds to DGLAP evolution with the relative number of
    running flavors, and the vertical bridges are the perturbative matchings
    between two different "adjacent" schemes.

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
