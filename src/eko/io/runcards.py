"""Structures to hold runcards information.

All energy scales in the runcards should be saved linearly, not the squared
value, for consistency.
Squares are consistenly taken inside.

"""
from dataclasses import dataclass, fields
from math import nan
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt

from eko.thresholds import ThresholdsAtlas

from .. import basis_rotation as br
from .. import interpolation, msbar_masses
from .. import version as vmod
from ..couplings import couplings_mod_ev
from ..quantities import heavy_quarks as hq
from ..quantities.heavy_quarks import (
    HeavyQuarkMasses,
    MatchingRatios,
    MatchingScales,
    QuarkMassScheme,
    scales_from_ratios,
)
from .dictlike import DictLike
from .types import (
    CouplingsRef,
    EvolutionMethod,
    FlavorsNumber,
    IntrinsicFlavors,
    InversionMethod,
    Order,
    RawCard,
    ScaleVariationsMethod,
    T,
    Target,
)


@dataclass
class TheoryCard(DictLike):
    """Represent theory card content."""

    order: Order
    """Perturbative order tuple, ``(QCD, QED)``."""
    couplings: CouplingsRef
    """Couplings configuration."""
    num_flavs_init: Optional[FlavorsNumber]
    r"""Number of active flavors at fitting scale.

    I.e. :math:`n_{f,\text{ref}}(\mu^2_0)`, formerly called ``nf0``.

    """
    num_flavs_max_pdf: FlavorsNumber
    """Maximum number of quark PDFs."""
    intrinsic_flavors: IntrinsicFlavors
    """List of intrinsic quark PDFs."""
    quark_masses: HeavyQuarkMasses
    """List of heavy quark masses."""
    quark_masses_scheme: QuarkMassScheme
    """Scheme used to specify heavy quark masses."""
    matching: MatchingRatios
    """Matching scale of heavy quark masses"""
    xif: float
    """Ratio between factorization scale and process scale."""

    @property
    def matching_scales(self) -> MatchingScales:
        """Compute matching scales."""
        return scales_from_ratios(self.matching, self.quark_masses)


def masses(theory: TheoryCard, evmeth: EvolutionMethod):
    """Compute masses in the chosen scheme."""
    if theory.quark_masses_scheme is QuarkMassScheme.MSBAR:
        return msbar_masses.compute(
            theory.quark_masses,
            theory.couplings,
            theory.order,
            couplings_mod_ev(evmeth),
            np.power(theory.matching, 2.0),
            xif2=theory.xif**2,
        ).tolist()
    if theory.quark_masses_scheme is QuarkMassScheme.POLE:
        return [mq.value**2 for mq in theory.quark_masses]

    raise ValueError(f"Unknown mass scheme '{theory.quark_masses_scheme}'")


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
class Rotations(DictLike):
    """Rotations related configurations.

    Here "Rotation" is intended in a broad sense: it includes both rotations in
    flavor space (labeled with suffix `pids`) and in :math:`x`-space (labeled
    with suffix `grid`).
    Rotations in :math:`x`-space correspond to reinterpolate the result on a
    different basis of polynomials.

    """

    xgrid: interpolation.XGrid
    """Internal momentum fraction grid."""
    _targetgrid: Optional[interpolation.XGrid] = None
    _inputgrid: Optional[interpolation.XGrid] = None
    _targetpids: Optional[npt.NDArray] = None
    _inputpids: Optional[npt.NDArray] = None

    def __post_init__(self):
        """Adjust types when loaded from serialized object."""
        for attr in ("xgrid", "_inputgrid", "_targetgrid"):
            value = getattr(self, attr)
            if value is None:
                continue
            if isinstance(value, (np.ndarray, list)):
                setattr(self, attr, interpolation.XGrid(value))
            elif not isinstance(value, interpolation.XGrid):
                setattr(self, attr, interpolation.XGrid.load(value))

    @property
    def pids(self):
        """Internal flavor basis, used for computation."""
        return np.array(br.flavor_basis_pids)

    @property
    def inputpids(self) -> npt.NDArray:
        """Provide pids expected on the input PDF."""
        if self._inputpids is None:
            return self.pids
        return self._inputpids

    @inputpids.setter
    def inputpids(self, value):
        self._inputpids = value

    @property
    def targetpids(self) -> npt.NDArray:
        """Provide pids corresponding to the output PDF."""
        if self._targetpids is None:
            return self.pids
        return self._targetpids

    @targetpids.setter
    def targetpids(self, value):
        self._targetpids = value

    @property
    def inputgrid(self) -> interpolation.XGrid:
        """Provide :math:`x`-grid expected on the input PDF."""
        if self._inputgrid is None:
            return self.xgrid
        return self._inputgrid

    @inputgrid.setter
    def inputgrid(self, value: interpolation.XGrid):
        self._inputgrid = value

    @property
    def targetgrid(self) -> interpolation.XGrid:
        """Provide :math:`x`-grid corresponding to the output PDF."""
        if self._targetgrid is None:
            return self.xgrid
        return self._targetgrid

    @targetgrid.setter
    def targetgrid(self, value: interpolation.XGrid):
        self._targetgrid = value

    @classmethod
    def from_dict(cls, dictionary: dict):
        """Deserialize rotation.

        Load from full state, but with public names.

        """
        d = dictionary.copy()
        for f in fields(cls):
            if f.name.startswith("_"):
                d[f.name] = d.pop(f.name[1:])
        return cls._from_dict(d)

    @property
    def raw(self):
        """Serialize rotation.

        Pass through interfaces, access internal values but with a public name.

        """
        d = self._raw()
        for key in d.copy():
            if key.startswith("_"):
                d[key[1:]] = d.pop(key)

        return d


@dataclass
class OperatorCard(DictLike):
    """Operator Card info."""

    mu0: float
    """Initial scale."""
    mugrid: List[Target]
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
        if "QrefQED" not in old:
            qedref = nan
        else:
            qedref = old["QrefQED"]
        new["couplings"] = dict(
            alphas=(old["alphas"], old["Qref"]),
            alphaem=(alphaem, qedref),
            num_flavs_ref=old["nfref"],
            max_num_flavs=old["MaxNfAs"],
        )
        new["num_flavs_init"] = old["nf0"]
        new["num_flavs_max_pdf"] = old["MaxNfPdf"]
        intrinsic = []
        for idx, q in enumerate(hq.FLAVORS):
            if old.get(f"i{q}".upper()) == 1:
                intrinsic.append(idx + 4)
        new["intrinsic_flavors"] = intrinsic
        new["matching"] = self.heavies("k%sThr", old)
        new["quark_masses_scheme"] = old["HQ"]
        ms = self.heavies("m%s", old)
        mus = self.heavies("Qm%s", old)
        if old["HQ"] == "POLE":
            new["quark_masses"] = [[m, nan] for m in ms]
        elif old["HQ"] == "MSBAR":
            new["quark_masses"] = [[m, mu] for m, mu in zip(ms, mus)]
        else:
            raise ValueError()

        new["xif"] = old["XIF"]

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
            mugrid = np.sqrt(mu2grid)
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
    tc = ThresholdsAtlas(
        masses=(np.array(masses) ** 2).tolist(),
        thresholds_ratios=(np.array(matching_ratios) ** 2).tolist(),
    )
    return [(mu, tc.nf(mu**2)) for mu in mugrid]
