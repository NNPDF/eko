"""Types and quantities related to theory couplings."""

import dataclasses
import enum

from ..io.dictlike import DictLike
from ..io.types import EvolutionPoint as EPoint
from ..io.types import ReferenceRunning, Scalar

Coupling = Scalar
CouplingRef = ReferenceRunning[Coupling]


@dataclasses.dataclass
class CouplingsInfo(DictLike):
    """Reference values for coupling constants.

    Also includes further information, defining the run of the
    couplings.
    """

    alphas: Coupling
    alphaem: Coupling
    ref: EPoint
    em_running: bool = False

    @property
    def values(self):
        """Collect only couplings values."""
        return (self.alphas, self.alphaem)


# TODO: upgrade the following to StrEnum (requires py>=3.11) with that, it is
# possible to replace all non-alias right sides with calls to enum.auto()


class CouplingEvolutionMethod(enum.Enum):
    """Beta functions solution method."""

    EXACT = "exact"
    EXPANDED = "expanded"
