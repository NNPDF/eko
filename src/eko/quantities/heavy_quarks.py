"""Heavy quarks related quantities."""
import dataclasses
import enum

from ..io.dictlike import DictLike
from ..io.types import LinearScale, reference_running


def _heavy_quark(quarkattr):
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


QuarkMass = LinearScale
QuarkMassRef = reference_running(QuarkMass)
HeavyQuarkMasses = _heavy_quark(QuarkMassRef)
MatchingScale = LinearScale
MatchingScales = _heavy_quark(MatchingScale)


# TODO: upgrade all the following to StrEnum (requires py>=3.11)
# with that, it is possible to replace all non-alias right sides with calls to
# enum.auto()


class QuarkMassScheme(enum.Enum):
    """Scheme to define heavy quark masses."""

    MSBAR = "msbar"
    POLE = "pole"
