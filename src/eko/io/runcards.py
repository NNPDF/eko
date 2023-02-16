"""Structures to hold runcards information.

All energy scales in the runcards should be saved linearly, not the squared
value, for consistency.
Squares are consistenly taken inside.

"""
from dataclasses import dataclass
from math import nan
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from .. import basis_rotation as br
from .. import interpolation
from .. import version as vmod
from .dictlike import DictLike
from .types import (
    CouplingsRef,
    EvolutionMethod,
    FlavorsNumber,
    HeavyQuarkMasses,
    IntrinsicFlavors,
    InversionMethod,
    MatchingScales,
    Order,
    QuarkMassSchemes,
    RawCard,
    ScaleVariationsMethod,
    T,
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
    quark_masses_scheme: QuarkMassSchemes
    """Scheme used to specify heavy quark masses."""
    matching: MatchingScales
    """Matching scale of heavy quark masses"""
    xif: float
    """Ratio between factorization scale and process scale."""


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
    scvar_method: Optional[ScaleVariationsMethod]
    """"""
    inversion_method: Optional[InversionMethod]
    """Which method to use for backward matching conditions."""
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
    """Momentum fraction internal grid."""
    pids: npt.NDArray
    """Array of integers, corresponding to internal PIDs."""
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


@dataclass
class OperatorCard(DictLike):
    """Operator Card info."""

    mu0: float
    """Initial scale."""

    # collections
    configs: Configs
    """Specific configuration to be used during the calculation of these operators."""
    debug: Debug
    """Debug configurations."""
    rotations: Rotations
    """Rotations configurations.

    The operator card will only contain the interpolation xgrid and the pids.

    """

    # TODO: drop legacy compatibility, only linear scales in runcards, such
    # that we will always avoid taking square roots, and we are consistent with
    # the other scales
    _mugrid: Optional[npt.NDArray] = None
    _mu2grid: Optional[npt.NDArray] = None
    """Array of final scales."""

    # optional
    eko_version: str = vmod.__version__
    """Version of EKO package first used for generation."""

    # a few properties, for ease of use and compatibility
    @property
    def mu20(self):
        """Squared value of initial scale."""
        return self.mu0**2

    @property
    def mugrid(self):
        """Only setter enabled, access only to :attr:`mu2grid`."""
        raise ValueError("Use mu2grid")

    @mugrid.setter
    def mugrid(self, value):
        """Set scale grid with linear values."""
        self._mugrid = value
        self._mu2grid = None

    @property
    def mu2grid(self):
        """Grid of squared final scales."""
        if self._mugrid is not None:
            return self._mugrid**2
        if self._mu2grid is not None:
            return self._mu2grid

        raise RuntimeError("Mu2 grid has not been initialized")

    @mu2grid.setter
    def mu2grid(self, value):
        """Set scale grid with quadratic values."""
        self._mugrid = None
        self._mu2grid = value


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
    HEAVY = "cbt"

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

        def heavies(pattern: str):
            return {q: old[pattern % q] for q in self.HEAVY}

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
        for idx, q in enumerate(self.HEAVY):
            if old.get(f"i{q}".upper()) == 1:
                intrinsic.append(idx + 4)
        new["intrinsic_flavors"] = intrinsic
        new["matching"] = heavies("k%sThr")
        new["quark_masses_scheme"] = old["HQ"]
        ms = heavies("m%s")
        mus = heavies("Qm%s")
        if old["HQ"] == "POLE":
            new["quark_masses"] = {q: (ms[q], nan) for q in self.HEAVY}
        elif old["HQ"] == "MSBAR":
            new["quark_masses"] = {q: (ms[q], mus[q]) for q in self.HEAVY}
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
        new["_mu2grid"] = old["Q2grid"]

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

        new["rotations"] = {}
        new["rotations"]["pids"] = old.get("pids", br.flavor_basis_pids)
        new["rotations"]["xgrid"] = old["interpolation_xgrid"]
        for basis in ("inputgrid", "targetgrid", "inputpids", "targetpids"):
            new["rotations"][f"_{basis}"] = old[basis]

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
