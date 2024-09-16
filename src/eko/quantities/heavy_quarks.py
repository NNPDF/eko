"""Heavy quarks related quantities."""

import enum
from dataclasses import dataclass
from typing import Generic, List, Sequence, TypeVar

import numpy as np

from ..io.dictlike import DictLike
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
        return self[1]

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


# TODO: upgrade the following to StrEnum (requires py>=3.11) with that, it is
# possible to replace all non-alias right sides with calls to enum.auto()


class QuarkMassScheme(enum.Enum):
    """Scheme to define heavy quark masses."""

    MSBAR = "msbar"
    POLE = "pole"


@dataclass
class HeavyInfo(DictLike):
    """Collect information about heavy quarks.

    This is meant to be used mainly as a theory card section, and to be passed
    around when all or a large part of this information is required.

    Note
    ----
    All scales and ratios in this structure are linear, so you can consider
    them as quantities in :math:`GeV` or ratios of them.
    """

    masses: HeavyQuarkMasses
    """List of heavy quark masses."""
    masses_scheme: QuarkMassScheme
    """Scheme used to specify heavy quark masses."""
    matching_ratios: MatchingRatios
    """Matching scale of heavy quark masses."""

    @property
    def squared_ratios(self) -> List[float]:
        """Squared ratios of matching scales."""
        return np.power(self.matching_ratios, 2.0).tolist()
