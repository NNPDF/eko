"""Operators bases."""

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import numpy.typing as npt

from .. import basis_rotation as br
from .. import interpolation
from .dictlike import DictLike


@dataclass
class Bases(DictLike):
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
