# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class Operator:
    alphas: float
    operators: np.ndarray
    operator_errors: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)


@dataclass
class Debug:
    skip_singlet: bool
    skip_non_singlet: bool


@dataclass
class Configs:
    ev_op_max_order: int
    ev_op_iterations: int
    interpolation_polynomial_degree: int
    interpolation_is_log: bool
    backward_inversion: str


@dataclass
class EKO:
    """
    Wrapper for the output to help with application
    to PDFs and dumping to file.
    """

    xgrid: np.ndarray
    targetgrid: np.ndarray
    inputgrid: np.ndarray
    targetpids: np.ndarray
    inputpids: np.ndarray
    Q02: float
    Q2grid: Dict[float, Operator]
    # _operators: Dict[float, Optional[Operator]]
    configs: Configs
    debug: Debug

    # @property
    # def Q2grid(self):
    #    return np.ndarray(self._operators.keys())

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)
