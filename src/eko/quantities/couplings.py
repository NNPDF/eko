"""Types and quantities related to theory couplings."""
import dataclasses
import enum
from typing import Optional

from ..io.dictlike import DictLike
from ..io.types import FlavorsNumber, LinearScale, ReferenceRunning, Scalar

Coupling = Scalar
CouplingRef = ReferenceRunning[Coupling]


@dataclasses.dataclass
class CouplingsInfo(DictLike):
    """Reference values for coupling constants.

    Also includes further information, defining the run of the couplings.

    """

    alphas: Coupling
    alphaem: Coupling
    scale: LinearScale
    max_num_flavs: FlavorsNumber
    num_flavs_ref: Optional[FlavorsNumber]
    r"""Number of active flavors at strong coupling reference scale.

    I.e. :math:`n_{f,\text{ref}}(\mu^2_{\text{ref}})`, formerly called
    ``nfref``.

    """
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
