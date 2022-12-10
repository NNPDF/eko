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

    inspec.isclass(RunningReference)  # False
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
from math import nan
from typing import Any, Dict

from .dictlike import DictLike

T = typing.TypeVar("T")
Quantity = typing.TypeVar("Quantity", bound=typing.Union[int, float])


def reference_running(quantity: typing.Type[Quantity]):
    """Generate running quantities reference point classes.

    The motivation for dynamic generation is provided in module docstring.

    """

    @dataclasses.dataclass
    class ReferenceRunning(DictLike):
        value: quantity
        scale: float

    return ReferenceRunning


IntRef = reference_running(int)
FloatRef = reference_running(float)

Order = typing.Tuple[int, int]


@dataclasses.dataclass
class CouplingsRef(DictLike):
    """Reference values for coupling constants."""

    alphas: FloatRef
    alphaem: FloatRef

    def __post_init__(self):
        """Validate couplings.

        If they are both running, they have to be defined at the same scale.

        Usually :attr:`alphaem` is not running, thus its scale is set to nan.

        """
        assert self.alphas.scale == self.alphaem.scale or self.alphaem.scale is nan

    @property
    def values(self):
        """Collect only couplings values."""
        return (self.alphas.value, self.alphaem.value)


FlavorsNumber = int
FlavorIndex = int
IntrinsicFlavors = typing.List[FlavorIndex]


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

        @classmethod
        def from_dict(cls, dictionary):
            if dataclasses.is_dataclass(quarkattr):
                return cls(**{k: quarkattr(*v) for k, v in dictionary.items()})
            else:
                return cls(**dictionary)

    return HeavyQuarks


QuarkMassRef = FloatRef
HeavyQuarkMasses = heavy_quark(QuarkMassRef)
MatchingScale = float
MatchingScales = heavy_quark(MatchingScale)


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
