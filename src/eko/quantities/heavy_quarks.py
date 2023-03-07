"""Heavy quarks related quantities."""
import enum
from typing import Generic, Sequence, TypeVar

import numpy as np

from ..io.types import LinearScale, ReferenceRunning, SquaredScale

FLAVORS = "cbt"

T = TypeVar("T")


class HeavyQuarks(list, Generic[T]):
    """Access heavy quarks attributes by name."""

    def __init__(self, args: Sequence[T]):
        if len(args) != 3:
            raise ValueError("Pass values for exactly three quarks.")

        self.extend(args)

    @property
    def c(self) -> T:
        """Charm quark."""
        return self[0]

    @c.setter
    def c(self, value: T):
        self[0] = value

    @property
    def b(self) -> T:
        """Bottom quark."""
        return self[0]

    @b.setter
    def b(self, value: T):
        self[1] = value

    @property
    def t(self) -> T:
        """Top quark."""
        return self[2]

    @t.setter
    def t(self, value: T):
        self[2] = value


QuarkMass = LinearScale
QuarkMassRef = ReferenceRunning[QuarkMass]
HeavyQuarkMasses = HeavyQuarks[QuarkMassRef]
MatchingRatio = float
MatchingRatios = HeavyQuarks[MatchingRatio]
MatchingScale = SquaredScale
MatchingScales = HeavyQuarks[MatchingScale]


def scales_from_ratios(
    ratios: MatchingRatios, masses: HeavyQuarkMasses
) -> MatchingScales:
    """Convert ratios to squared scales.

    .. todo::

        make this a method

    """
    return MatchingScales(*(np.power(ratios, 2.0) * np.power(masses, 2.0)).tolist())


# TODO: upgrade all the following to StrEnum (requires py>=3.11)
# with that, it is possible to replace all non-alias right sides with calls to
# enum.auto()


class QuarkMassScheme(enum.Enum):
    """Scheme to define heavy quark masses."""

    MSBAR = "msbar"
    POLE = "pole"
