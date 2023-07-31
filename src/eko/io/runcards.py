"""Structures to hold runcards information.

All energy scales in the runcards should be saved linearly, not the squared
value, for consistency.
Squares are consistenly taken inside.

"""
from dataclasses import dataclass
from math import nan
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt

from .. import basis_rotation as br
from .. import interpolation, msbar_masses
from .. import version as vmod
from ..couplings import couplings_mod_ev
from ..matchings import Atlas, nf_default
from ..quantities import heavy_quarks as hq
from ..quantities.couplings import CouplingsInfo
from ..quantities.heavy_quarks import HeavyInfo, QuarkMassScheme
from .dictlike import DictLike
from .types import EvolutionMethod
from .types import EvolutionPoint as EPoint
from .types import (
    InversionMethod,
    N3LOAdVariation,
    Order,
    RawCard,
    ScaleVariationsMethod,
    SquaredScale,
    T,
)


# TODO: add frozen
@dataclass
class TheoryCard(DictLike):
    """Represent theory card content."""

    order: Order
    """Perturbative order tuple, ``(QCD, QED)``."""
    couplings: CouplingsInfo
    """Couplings configuration."""
    heavy: HeavyInfo
    """Heavy quarks related information."""
    xif: float
    """Ratio between factorization scale and process scale."""
    n3lo_ad_variation: N3LOAdVariation
    """|N3LO| anomalous dimension variation: ``(gg_var, gq_var, qg_var, qq_var)``."""


@dataclass
class Debug(DictLike):
    """Debug configurations."""

    skip_singlet: bool = False
    """Whether to skip QCD singlet computation."""
    skip_non_singlet: bool = False
    """Whether to skip QCD non-singlet computation."""


@dataclass
class Configs(DictLike):
    """Solution specific configurations."""

    evolution_method: EvolutionMethod
    """Evolution mode."""
    ev_op_max_order: Order
    """Maximum order to use in U matrices expansion.
    Used only in ``perturbative`` solutions.
    """
    ev_op_iterations: int
    """Number of intervals in which to break the global path."""
    scvar_method: Optional[ScaleVariationsMethod]
    """Scale variation method."""
    inversion_method: Optional[InversionMethod]
    """Which method to use for backward matching conditions."""
    interpolation_polynomial_degree: int
    """Degree of elements of the intepolation polynomial basis."""
    interpolation_is_log: bool
    r"""Whether to use polynomials in :math:`\log(x)`.
    If `false`, polynomials are in :math:`x`.
    """
    polarized: bool
    """If `true` do polarized evolution."""
    time_like: bool
    """If `true` do time-like evolution."""
    n_integration_cores: int = 1
    """Number of cores used to parallelize integration."""


@dataclass
class OperatorCard(DictLike):
    """Operator Card info."""

    mu0: float
    """Initial scale."""
    mugrid: List[EPoint]
    xgrid: interpolation.XGrid
    """Momentum fraction internal grid."""

    # collections
    configs: Configs
    """Specific configuration to be used during the calculation of these operators."""
    debug: Debug
    """Debug configurations."""

    # optional
    eko_version: str = vmod.__version__
    """Version of EKO package first used for generation."""

    # a few properties, for ease of use and compatibility
    @property
    def mu20(self):
        """Squared value of initial scale."""
        return self.mu0**2

    @property
    def mu2grid(self) -> npt.NDArray:
        """Grid of squared final scales."""
        return np.array([mu for mu, _ in self.mugrid]) ** 2

    @property
    def evolgrid(self) -> List[EPoint]:
        """Grid of squared final scales."""
        return [(mu**2, nf) for mu, nf in self.mugrid]

    @property
    def pids(self):
        """Internal flavor basis, used for computation."""
        return np.array(br.flavor_basis_pids)


Card = Union[TheoryCard, OperatorCard]


@dataclass
class Legacy:
    """Upgrade legacy runcards."""

    theory: RawCard
    operator: RawCard

    MOD_EV2METHOD = {
        "EXA": "iterate-exact",
        "EXP": "iterate-expanded",
        "TRN": "truncated",
    }

    @staticmethod
    def heavies(pattern: str, old_th: dict):
        """Retrieve a set of values for all heavy flavors."""
        return [old_th[pattern % q] for q in hq.FLAVORS]

    @staticmethod
    def fallback(*args: T, default: Optional[T] = None) -> T:
        """Return the first not None argument."""
        for arg in args:
            if arg is not None:
                return arg

        if default is None:
            return args[-1]
        return default

    @property
    def new_theory(self):
        """Build new format theory runcard."""
        old = self.theory
        new = {}

        new["order"] = [old["PTO"] + 1, old["QED"]]
        alphaem = self.fallback(old.get("alphaqed"), old.get("alphaem"), default=0.0)
        if "Qedref" in old:
            em_running = bool(np.isclose(old["Qedref"], old["Qref"]))
        else:
            em_running = False
        ms = self.heavies("m%s", old)
        ks = self.heavies("k%sThr", old)
        new["couplings"] = dict(
            alphas=old["alphas"],
            alphaem=alphaem,
            em_running=em_running,
            scale=old["Qref"],
            num_flavs_ref=old["nfref"],
        )
        new["heavy"] = {
            "num_flavs_init": nf_default(old["Q0"] ** 2.0, default_atlas(ms, ks))
            if old["nf0"] is None
            else old["nf0"],
            "num_flavs_max_pdf": old["MaxNfPdf"],
            "matching_ratios": self.heavies("k%sThr", old),
            "masses_scheme": old["HQ"],
        }
        intrinsic = []
        for idx, q in enumerate(hq.FLAVORS):
            if old.get(f"i{q}".upper()) == 1:
                intrinsic.append(idx + 4)
        new["heavy"]["intrinsic_flavors"] = intrinsic
        if old["HQ"] == "POLE":
            new["heavy"]["masses"] = [[m, nan] for m in ms]
        elif old["HQ"] == "MSBAR":
            mus = self.heavies("Qm%s", old)
            new["heavy"]["masses"] = [[m, mu] for m, mu in zip(ms, mus)]
        else:
            raise ValueError(f"Unknown mass scheme '{old['HQ']}'")

        new["xif"] = old["XIF"]
        new["n3lo_ad_variation"] = old.get("n3lo_ad_variation", (0, 0, 0, 0))

        return TheoryCard.from_dict(new)

    @property
    def new_operator(self):
        """Build new format operator runcard."""
        old = self.operator
        old_th = self.theory
        new = {}

        new["mu0"] = old_th["Q0"]
        if "mugrid" in old:
            mugrid = old["mugrid"]
        else:
            mu2grid = old["Q2grid"] if "Q2grid" in old else old["mu2grid"]
            mugrid = np.sqrt(mu2grid).tolist()
        new["mugrid"] = flavored_mugrid(
            mugrid,
            list(self.heavies("m%s", old_th)),
            list(self.heavies("k%sThr", old_th)),
        )

        new["configs"] = {}
        evmod = old_th["ModEv"]
        new["configs"]["evolution_method"] = self.MOD_EV2METHOD.get(evmod, evmod)
        new["configs"]["inversion_method"] = old_th.get(
            "backward_inversion", "expanded"
        )
        new["configs"]["scvar_method"] = old_th.get("ModSV", "expanded")
        for k in (
            "interpolation_polynomial_degree",
            "interpolation_is_log",
            "ev_op_iterations",
            "n_integration_cores",
            "polarized",
            "time_like",
        ):
            new["configs"][k] = old[k]
        max_order = old["ev_op_max_order"]
        if isinstance(max_order, int):
            new["configs"]["ev_op_max_order"] = [max_order, old_th["QED"]]

        new["debug"] = {}
        lpref = len("debug_")
        for k in ("debug_skip_non_singlet", "debug_skip_singlet"):
            new["debug"][k[lpref:]] = old[k]

        new["xgrid"] = old["interpolation_xgrid"]

        return OperatorCard.from_dict(new)


def update(theory: Union[RawCard, TheoryCard], operator: Union[RawCard, OperatorCard]):
    """Update legacy runcards.

    This function is mainly defined for compatibility with the old interface.
    Prefer direct usage of :class:`Legacy` in new code.

    Consecutive applications of this function yield identical results::

        cards = update(theory, operator)
        assert update(*cards) == cards

    """
    if isinstance(theory, TheoryCard) or isinstance(operator, OperatorCard):
        # if one is not a dict, both have to be new cards
        assert isinstance(theory, TheoryCard)
        assert isinstance(operator, OperatorCard)
        return theory, operator

    cards = Legacy(theory, operator)
    return cards.new_theory, cards.new_operator


def default_atlas(masses: list, matching_ratios: list):
    r"""Create default landscape.

    This method should not be used to write new runcards, but rather to have a
    consistent default for comparison with other softwares and existing PDF
    sets.
    There is no one-to-one relation between number of running flavors and final
    scales, unless matchings are all applied. But this is a custom choice,
    since it is possible to have PDFs in different |FNS| at the same scales.

    """
    matchings = (np.array(masses) * np.array(matching_ratios)) ** 2
    return Atlas(matchings.tolist(), (0.0, 0))


def flavored_mugrid(mugrid: list, masses: list, matching_ratios: list):
    r"""Upgrade :math:`\mu^2` grid to contain also target number flavors.

    It determines the number of flavors for the PDF set at the target scale,
    inferring it according to the specified scales.

    This method should not be used to write new runcards, but rather to have a
    consistent default for comparison with other softwares and existing PDF
    sets.
    There is no one-to-one relation between number of running flavors and final
    scales, unless matchings are all applied. But this is a custom choice,
    since it is possible to have PDFs in different |FNS| at the same scales.

    """
    atlas = default_atlas(masses, matching_ratios)
    return [(mu, nf_default(mu**2, atlas)) for mu in mugrid]


# TODO: move to a more suitable place
def masses(theory: TheoryCard, evmeth: EvolutionMethod) -> List[SquaredScale]:
    """Compute masses in the chosen scheme."""
    if theory.heavy.masses_scheme is QuarkMassScheme.MSBAR:
        return msbar_masses.compute(
            theory.heavy.masses,
            theory.couplings,
            theory.order,
            couplings_mod_ev(evmeth),
            np.power(theory.heavy.matching_ratios, 2.0).tolist(),
            xif2=theory.xif**2,
        ).tolist()
    if theory.heavy.masses_scheme is QuarkMassScheme.POLE:
        return [mq.value**2 for mq in theory.heavy.masses]

    raise ValueError(f"Unknown mass scheme '{theory.heavy.masses_scheme}'")
