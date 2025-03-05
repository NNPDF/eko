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
    """Reference values for coupling constants."""

    alphas: Coupling
    r"""Reference |QCD| coupling :math:`\alpha_s(\mu_R^0)^{(n_f^0)}`.

    Note
    ----
    We refer to :math:`\alpha_s` here instead
    of :math:`a_s = \alpha_s/(4\pi)` as we do otherwise.
    """
    alphaem: Coupling
    r"""Reference |QED| coupling :math:`\alpha_{em}(\mu_R^0)^{(n_f^0)}`."""
    ref: EPoint
    r"""Reference evolution point :math:`(\mu_R^0,n_f^0)`"""
    em_running: bool = False
    r"""If set, activates the |QED| running."""

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
