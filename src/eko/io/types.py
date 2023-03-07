"""Common type definitions, only used for static analysis."""
import dataclasses
import enum
import typing
from math import isnan
from typing import Any, Dict, Generic, Optional, TypeVar

from .dictlike import DictLike

T = typing.TypeVar("T")

# Energy scales
# -------------

Scale = float

LinearScale = Scale
SquaredScale = Scale
# TODO: replace with (requires py>=3.9)
#  LinearScale = Annotated[Scale, 1]
#  SquaredScale = Annotated[Scale, 2]


# Flavors
# -------

Order = typing.Tuple[int, int]
FlavorsNumber = int
FlavorIndex = int
IntrinsicFlavors = typing.List[FlavorIndex]


# Scale functions
# ---------------

T = TypeVar("T")


class ReferenceRunning(list, Generic[T]):
    """Running quantities reference point.

    To simplify serialization, the class is just storing the content as a list,
    but:

    - it is constructed with a ``Running(T, Scale)`` signature
    - it should always be used through the property accessors, rather then
      using the list itself

    """

    @classmethod
    def typed(cls, value: T, scale: Scale):
        return cls([value, scale])

    @property
    def value(self) -> T:
        """Reference value, given at a specified scale."""
        return self[0]

    @value.setter
    def value(self, value: T):
        self[0] = value

    @property
    def scale(self) -> Scale:
        """Reference scale, at which the value of the function is given."""
        return self[1]

    @scale.setter
    def scale(self, value: Scale):
        self[1] = value


FlavNumRef = ReferenceRunning[FlavorsNumber]
ScalarRef = ReferenceRunning[float]
LinearScaleRef = ReferenceRunning[LinearScale]


# Numerical methods
# -----------------

# TODO: upgrade all the following to StrEnum (requires py>=3.11)
# with that, it is possible to replace all non-alias right sides with calls to
# enum.auto()


class EvolutionMethod(enum.Enum):
    """DGLAP solution method."""

    ITERATE_EXACT = "iterate-exact"
    ITERATE_EXPANDED = "iterate-expanded"
    PERTURBATIVE_EXACT = "perturbative-exact"
    PERTURBATIVE_EXPANDED = "perturbative-expanded"
    TRUNCATED = "truncated"
    ORDERED_TRUNCATED = "ordered-truncated"
    DECOMPOSE_EXACT = "decompose-exact"
    DECOMPOSE_EXPANDED = "decompose-expanded"


class CouplingEvolutionMethod(enum.Enum):
    """Beta functions solution method."""

    EXACT = "exact"
    EXPANDED = "expanded"


class ScaleVariationsMethod(enum.Enum):
    """Method used to account for factorization scale variation."""

    EXPONENTIATED = "exponentiated"
    EXPANDED = "expanded"


class InversionMethod(enum.Enum):
    """Method used to invert the perturbative matching conditions."""

    EXACT = "exact"
    EXPANDED = "expanded"


RawCard = Dict[str, Any]


# Couplings
# ---------


@dataclasses.dataclass
class CouplingsRef(DictLike):
    """Reference values for coupling constants."""

    alphas: ScalarRef
    alphaem: ScalarRef
    max_num_flavs: FlavorsNumber
    num_flavs_ref: Optional[FlavorsNumber]
    r"""Number of active flavors at strong coupling reference scale.

    I.e. :math:`n_{f,\text{ref}}(\mu^2_{\text{ref}})`, formerly called
    ``nfref``.

    """

    def __post_init__(self):
        """Validate couplings.

        If they are both running, they have to be defined at the same scale.

        Usually :attr:`alphaem` is not running, thus its scale is set to nan.

        """
        assert self.alphas.scale == self.alphaem.scale or isnan(self.alphaem.scale)

    @property
    def values(self):
        """Collect only couplings values."""
        return (self.alphas.value, self.alphaem.value)
