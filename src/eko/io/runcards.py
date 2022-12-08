"""Structures to hold runcards information."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from .. import basis_rotation as br
from .. import interpolation
from .. import version as vmod
from ..types import (
    CouplingConstants,
    EvolutionMethod,
    FlavorsNumber,
    FlavorsNumberRef,
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
from .dictlike import DictLike


@dataclass
class TheoryCard(DictLike):
    """Represent theory card content."""

    order: Order
    """Perturbatiive order tuple, ``(QCD, QED)``."""
    couplings: CouplingConstants
    """"""
    num_flavs_ref: FlavorsNumberRef
    r"""Number of active flavors at reference scale.

    This is the scale :math:`\mu^2_{\text{ref}}` appearing in
    :math:`n_{f,\text{ref}}(\mu^2_{\text{ref}})`.

    """
    num_flavs_init: FlavorsNumber
    """"""
    num_flavs_max_as: FlavorsNumber
    """"""
    intrinsic_flavors: IntrinsicFlavors
    """"""
    quark_masses: HeavyQuarkMasses
    """"""
    matching: MatchingScales
    """"""
    heavy_quark_masses: QuarkMassSchemes
    """Scheme used to specify heavy quark masses."""

    def validate(self) -> bool:
        """Validate attributes compatibility."""
        if self.heavy_quark_masses:
            return False
        return True

    @classmethod
    def from_dict(cls, card: dict):
        """Load from runcard raw content.

        Parameters
        ----------
        card: dict
            content of the theory runcard

        Returns
        -------
        TheoryCard
            the loaded instance

        """
        for entry in ["order", "quark_masses", "num_flavs_ref"]:
            card[entry] = tuple(card[entry])
        return super().from_dict(card)


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
    scvar_method: ScaleVariationsMethod
    """"""
    backward_inversion: InversionMethod
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
    def inputgrid(self, value):
        self._inputgrid = value

    @property
    def targetgrid(self) -> interpolation.XGrid:
        """Provide :math:`x`-grid corresponding to the output PDF."""
        if self._targetgrid is None:
            return self.xgrid
        return self._targetgrid

    @targetgrid.setter
    def targetgrid(self, value):
        self._targetgrid = value


@dataclass
class OperatorCard(DictLike):
    """Operator Card info."""

    mu0: float
    """Initial scale."""

    # collections
    configs: Configs
    """Specific configuration to be used during the calculation of these operators."""
    rotations: Rotations
    """Rotations configurations.

    The operator card will only contain the interpolation xgrid and the pids.

    """
    debug: Debug
    """Debug configurations."""

    mu2grid: npt.NDArray
    """Array of q2 points."""
    eko_version: Optional[str] = vmod.__version__
    """Perturbative order of the QCD and QED couplings."""

    @property
    def raw(self):
        """Return the raw dictionary to be dumped."""
        dictionary: RawCard = dict(
            mu0=self.mu0,
            mu2grid=self.mu2grid.tolist(),
            eko_version=self.eko_version,
        )
        dictionary["rotations"] = Rotations.from_dict(self.rotations).raw
        dictionary["configs"] = Configs.from_dict(self.configs).raw
        dictionary["debug"] = Debug.from_dict(self.debug).raw
        return dictionary


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
            return [old[pattern % q] for q in self.HEAVY]

        new["order"] = [old["PTO"] + 1, old["QED"]]
        alphaem = self.fallback(old.get("alphaqed"), old.get("alphaem"), default=0.0)
        new["couplings"] = [[old["alphas"], old["Qref"]], [alphaem, 0.0]]
        new["num_flavs_ref"] = (old["nfref"], old["Qref"])
        new["num_flavs_init"] = old["nf0"]
        new["num_flavs_max_as"] = old["MaxNfAs"]
        intrinsic = []
        for idx, q in enumerate(self.HEAVY):
            if old[f"i{q}".upper()] == 1:
                intrinsic.append(idx + 4)
        new["intrinsic_flavors"] = intrinsic
        new["matching"] = heavies("k%sThr")
        new["heavy_quark_masses"] = old["HQ"]
        if old["HQ"] == "POLE":
            new["quark_masses"] = heavies("m%s")
        elif old["HQ"] == "MSBAR":
            new["quark_masses"] = list(zip(heavies("m%s"), heavies("Qm%s")))
        else:
            raise ValueError()

        return TheoryCard.from_dict(new)

    @property
    def new_operator(self):
        """Build new format operator runcard."""
        old = self.operator
        old_th = self.theory
        new = {}

        new["mu0"] = old_th["Q0"]
        new["mugrid"] = np.sqrt(old["Q2grid"]).tolist()
        evmod = old_th["EvMod"]
        new["evolution_method"] = self.MOD_EV2METHOD.get(evmod, evmod)
        new["inversion_method"] = old_th["backward_inversion"]

        new["configs"] = {}
        for k in (
            "interpolation_polynomial_degree",
            "interpolation_is_log",
            "ev_op_iterations",
            "backward_inversion",
            "n_integration_cores",
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
            new["rotations"][basis] = old[basis]

        return OperatorCard.from_dict(new)
