"""Common type definitions, only used for static analysis."""
import enum
import typing
from typing import Any, Dict, List, Tuple

T = typing.TypeVar("T")

Quantity = typing.TypeVar("Quantity", bound=typing.Union[int, float])
RunningReference = Tuple[Quantity, float]
IntRef = RunningReference[int]
FloatRef = RunningReference[float]

Order = Tuple[int, int]
CouplingConstants = Tuple[FloatRef, FloatRef]
FlavorsNumber = int
FlavorsNumberRef = IntRef
FlavorIndex = int
IntrinsicFlavors = List[FlavorIndex]
QuarkMass = float
QuarkMassRef = typing.Union[FloatRef, QuarkMass]
HeavyQuarkMasses = Tuple[QuarkMassRef, QuarkMassRef, QuarkMassRef]
MatchingScale = float
MatchingScales = Tuple[MatchingScale, MatchingScale, MatchingScale]

# TODO: upgrade all the following to StrEnum, new in py3.11
# with that, it is possible to replace all non-alias right sides with calls to
# enum.auto()


class QuarkMassSchemes(enum.Enum):
    """Scheme to define heavy quark masses."""

    MSBAR = "msbar"
    POLE = "pole"


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


class ScaleVariationsMethod(enum.Enum):
    """Method used to account for factorization scale variation."""

    EXPONENTIATED = "exponentiated"
    EXPANDED = "expanded"


class InversionMethod(enum.Enum):
    """Method used to invert the perturbative matching conditions."""

    EXACT = "exact"
    EXPANDED = "expanded"


RawCard = Dict[str, Any]
