# -*- coding: utf-8 -*-
from dataclasses import dataclass, fields
from typing import Dict, Literal, Optional

import numpy as np


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
            dictionary[field.name] = (
                value if not isinstance(value, np.ndarray) else value.tolist()
            )


@dataclass
class Operator(DictLike):
    alphas: float
    operators: np.ndarray
    operator_errors: Optional[np.ndarray] = None


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

    xgrid: np.ndarray
    Q02: float
    _operators: Dict[float, Optional[Operator]]
    configs: Configs
    rotations: Rotations
    debug: Debug

    @property
    def Q2grid(self):
        return np.array(self._operators.keys())

    @classmethod
    def from_dict(cls, dictionary):
        return cls(
            xgrid=dictionary["xgrid"],
            Q02=dictionary["Q0"] ** 2,
            _operators={q2: None for q2 in dictionary["Q2grid"]},
            configs=Configs.from_dict(dictionary["configs"]),
            rotations=Rotations.from_dict(dictionary["rotations"]),
            debug=Debug.from_dict(dictionary.get("debug", {})),
        )

    @property
    def raw(self):
        return dict(
            xgrid=self.xgrid.tolist(),
            Q0=np.sqrt(self.Q02),
            Q2grid=self.Q2grid,
            configs=self.configs.raw,
            rotations=self.rotations.raw,
            debug=self.debug.raw,
        )
