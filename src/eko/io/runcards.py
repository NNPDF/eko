"""Structures to hold runcards information."""
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .. import interpolation
from .. import version as vmod
from .dictlike import DictLike


@dataclass
class TheoryCard(DictLike):
    """Represent theory card content."""

    order: Tuple[int, int]
    """Perturbatiive order tuple, ``(QCD, QED)``."""

    @classmethod
    def load(cls, card: dict):
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
        return cls(order=tuple(card["order"]))


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

    ev_op_max_order: Tuple[int]
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
    backward_inversion: Literal["exact", "expanded"]
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

    @property
    def targetpids(self) -> npt.NDArray:
        """Provide pids corresponding to the output PDF."""
        if self._targetpids is None:
            return self.pids
        return self._targetpids

    @property
    def inputgrid(self) -> interpolation.XGrid:
        """Provide :math:`x`-grid expected on the input PDF."""
        self.__post_init__()
        if self._inputgrid is None:
            return self.xgrid
        return self._inputgrid

    @property
    def targetgrid(self) -> interpolation.XGrid:
        """Provide :math:`x`-grid corresponding to the output PDF."""
        self.__post_init__()
        if self._targetgrid is None:
            return self.xgrid
        return self._targetgrid


@dataclass
class OperatorCard(DictLike):
    """Operator Card info."""

    Q0: float
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

    Q2grid: npt.NDArray
    """Array of q2 points."""
    eko_version: Optional[str] = vmod.__version__
    """Perturbative order of the QCD and QED couplings."""

    @property
    def raw(self):
        """Return the raw dictionary to be dumped."""
        dictionary: Dict[str, Any] = dict(
            Q0=float(self.Q0),
            Q2grid=self.Q2grid.tolist(),
            eko_version=self.eko_version,
        )
        dictionary["rotations"] = Rotations.from_dict(self.rotations).raw
        dictionary["configs"] = Configs.from_dict(self.configs).raw
        dictionary["debug"] = Debug.from_dict(self.debug).raw
        return dictionary
