"""Common type definitions, only used for static analysis."""

import enum
from typing import Any, Dict, Generic, List, Tuple, TypeVar

# Energy scales
# -------------

Scalar = float
Scale = float

LinearScale = Scale
SquaredScale = Scale
# TODO: replace with (requires py>=3.9)
#  LinearScale = Annotated[Scale, 1]
#  SquaredScale = Annotated[Scale, 2]


# Flavors
# -------

Order = Tuple[int, int]
FlavorsNumber = int
FlavorIndex = int
IntrinsicFlavors = List[FlavorIndex]
N3LOAdVariation = Tuple[int, int, int, int, int, int, int]
OperatorLabel = Tuple[int, int]

# Evolution coordinates
# ---------------------

EvolutionPoint = Tuple[Scale, FlavorsNumber]


# Scale functions
# ---------------

T = TypeVar("T")


class ReferenceRunning(list, Generic[T]):
    """Running quantities reference point.

    To simplify serialization, the class is just storing the content as a list,
    but:

    - it is constructed with a ``Running.typed(T, Scale)`` signature
    - it should always be used through the property accessors, rather then
      using the list itself
    """

    @classmethod
    def typed(cls, value: T, scale: Scale):
        """Define constructor from individual values.

        This is the preferred constructor for references, since respects
        the intended types of the values. It is not the default one only
        to simplify (de)serialization.
        """
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


class ScaleVariationsMethod(enum.Enum):
    """Method used to account for factorization scale variation."""

    EXPONENTIATED = "exponentiated"
    EXPANDED = "expanded"


class InversionMethod(enum.Enum):
    """Method used to invert the perturbative matching conditions."""

    EXACT = "exact"
    EXPANDED = "expanded"


RawCard = Dict[str, Any]
