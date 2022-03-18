# -*- coding: utf-8 -*-
from dataclasses import dataclass, fields
from typing import Dict, Literal, Optional

import numpy as np

from .. import interpolation
from .. import version as vmod


class DictLike:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)

    @property
    def raw(self):
        dictionary = {}
        for field in fields(self):
            value = getattr(self, field.name)

            # replace numpy arrays with lists
            if isinstance(value, np.ndarray):
                value = value.tolist()
            # replace numpy scalars with python ones
            elif isinstance(value, float):
                value = float(value)

            dictionary[field.name] = value

        return dictionary


@dataclass
class Operator(DictLike):
    alphas: float
    operator: np.ndarray
    error: Optional[np.ndarray] = None


@dataclass
class Debug(DictLike):
    skip_singlet: bool = False
    skip_non_singlet: bool = False


@dataclass
class Configs(DictLike):
    ev_op_max_order: int
    ev_op_iterations: int
    interpolation_polynomial_degree: int
    interpolation_is_log: bool
    backward_inversion: Literal["exact", "expanded"]


@dataclass
class Rotations(DictLike):
    targetgrid: Optional[np.ndarray] = None
    inputgrid: Optional[np.ndarray] = None
    targetpids: Optional[np.ndarray] = None
    inputpids: Optional[np.ndarray] = None


@dataclass
class EKO:
    """
    Wrapper for the output to help with application
    to PDFs and dumping to file.
    """

    xgrid: interpolation.XGrid
    Q02: float
    _operators: Dict[float, Optional[Operator]]
    configs: Configs
    rotations: Rotations
    debug: Debug
    version: str = vmod.__version__
    data_version: str = vmod.__data_version__

    def __iter__(self):
        return iter(self._operators)

    def __contains__(self, q2):
        return q2 in self._operators

    def __getitem__(self, q2):
        # TODO: autoload
        return self._operators[q2]

    def __setitem__(self, q2, op):
        # TODO: autodump
        if isinstance(op, dict):
            op = Operator.from_dict(op)
        if not isinstance(op, Operator):
            raise ValueError("Only operators can be stored.")
        self._operators[q2] = op

    def items(self):
        return self._operators.items()

    @property
    def Q2grid(self):
        return np.array(list(self._operators))

    @classmethod
    def from_dict(cls, dictionary):
        return cls(
            xgrid=interpolation.XGrid(dictionary["xgrid"]),
            Q02=dictionary["Q0"] ** 2,
            _operators={q2: None for q2 in dictionary["Q2grid"]},
            configs=Configs.from_dict(dictionary["configs"]),
            rotations=Rotations.from_dict(dictionary.get("rotations", {})),
            debug=Debug.from_dict(dictionary.get("debug", {})),
        )

    @classmethod
    def load(cls, path):
        return cls(
            xgrid=None,
            Q02=None,
            _operators={q2: None for q2 in ()},
            configs=None,
            rotations=None,
            debug=None,
        )

    @property
    def raw(self):
        return dict(
            xgrid=self.xgrid.tolist(),
            Q0=float(np.sqrt(self.Q02)),
            Q2grid=self.Q2grid,
            configs=self.configs.raw,
            rotations=self.rotations.raw,
            debug=self.debug.raw,
        )
