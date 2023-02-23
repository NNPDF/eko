"""Common type definitions, only used for static analysis.

Unfortunately, Python has still some discrepancies between runtime classes and
type hints, so it is better not to mix dataclasses and generics.

E.g. before it was implemented::

    @dataclasses.dataclass
    class RunningReference(DictLike, Generic[Quantity]):
        value: Quantity
        scale: float

but in this way it is not possible to determine that ``RunningReference`` is
subclassing ``DictLike``, indeed::

    inspect.isclass(RunningReference)       # False
    issubclass(RunningReference, DictLike)  # raise an error, since
                                            # RunningReference is not a class

Essentially classes can be used for type hints, but types are not all classes,
especially when they involve generics.

For this reason I prefer the less elegant dynamic generation, that seems to
preserve type hints.

"""
import dataclasses
import enum
import typing
from math import isnan
from typing import Any, Dict, NewType, Optional

from .dictlike import DictLike

T = typing.TypeVar("T")

# Energy scales
# -------------

Scale = NewType("Scale", float)

LinearScale = NewType("LinearScale", Scale)
SquaredScale = NewType("SquaredScale", Scale)
# TODO: replace with (requires py>=3.9)
#  LinearScale = Annotated[Scale, 1]
#  SquaredScale = Annotated[Scale, 2]


# Flavors
# -------

Order = NewType("Order", typing.Tuple[int, int])
FlavorsNumber = NewType("FlavorsNumber", int)
FlavorIndex = NewType("FlavorIndex", int)
IntrinsicFlavors = NewType("IntrinsicFlavors", typing.List[FlavorIndex])


# Scale functions
# ---------------


def reference_running(quantity: typing.Type[typing.Union[int, float]]):
    """Generate running quantities reference point classes.

    The motivation for dynamic generation is provided in module docstring.

    """

    @dataclasses.dataclass
    class ReferenceRunning(DictLike):
        value: quantity
        scale: Scale

    return ReferenceRunning


IntRef = reference_running(int)
FloatRef = reference_running(float)


# Heavy quarks
# ------------


def heavy_quark(quarkattr):
    """Generate heavy quark properties container classes.

    The motivation for dynamic generation is provided in module docstring.

    """

    @dataclasses.dataclass
    class HeavyQuarks(DictLike):
        """Access heavy quarks attributes by name."""

        c: quarkattr
        """Charm quark."""
        b: quarkattr
        """Bottom quark."""
        t: quarkattr
        """Top quark."""

        def __getitem__(self, key: int):
            """Allow access by index.

            Consequently it allows iteration and containing check.

            """
            return getattr(self, "cbt"[key])

    return HeavyQuarks


QuarkMass = NewType("QuarkMass", LinearScale)
QuarkMassRef = reference_running(QuarkMass)
HeavyQuarkMasses = heavy_quark(QuarkMassRef)
MatchingScale = NewType("MatchingScale", LinearScale)
MatchingScales = heavy_quark(MatchingScale)


# TODO: upgrade all the following to StrEnum (requires py>=3.11)
# with that, it is possible to replace all non-alias right sides with calls to
# enum.auto()


class QuarkMassSchemes(enum.Enum):
    """Scheme to define heavy quark masses."""

    MSBAR = "msbar"
    POLE = "pole"


# Numerical methods
# -----------------


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


RawCard = NewType("RawCard", Dict[str, Any])


# Couplings
# ---------


@dataclasses.dataclass
class CouplingsRef(DictLike):
    """Reference values for coupling constants."""

    alphas: FloatRef
    alphaem: FloatRef
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
