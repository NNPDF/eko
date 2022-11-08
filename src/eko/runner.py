"""Main application class of eko."""
import copy
import logging
from typing import Optional

import numpy as np

from . import compatibility, interpolation, msbar_masses
from .couplings import Couplings
from .evolution_operator.grid import OperatorGrid
from .output import EKO, Operator, manipulate
from .thresholds import ThresholdsAtlas

logger = logging.getLogger(__name__)


class Runner:
    """Represents a single input configuration.

    For details about the configuration, see :doc:`here </code/IO>`

    Attributes
    ----------
    setup : dict
        input configurations

    """

    banner = r"""
oooooooooooo oooo    oooo  \\ .oooooo.
`888'     `8 `888   .8P'  //////    `Y8b
 888          888  d8'   \\o\/////    888
 888oooo8     88888     \\\\/8/////   888
 888    "     888`88b.      888 ///   888
 888       o  888  `88b.    `88b //  d88'
o888ooooood8 o888o  o888o     `Y8bood8P'
"""

    def __init__(self, theory_card: dict, operators_card: dict):
        """Initialize runner.

        Parameters
        ----------
        theory_card: dict
            theory parameters and options
        operators_card: dict
            operator specific options

        """
        new_theory, new_operators = compatibility.update(theory_card, operators_card)

        # Store inputs
        self._theory = new_theory

        # setup basis grid
        bfd = interpolation.InterpolatorDispatcher(
            xgrid=interpolation.XGrid(
                new_operators["rotations"]["xgrid"],
                log=new_operators["configs"]["interpolation_is_log"],
            ),
            polynomial_degree=new_operators["configs"][
                "interpolation_polynomial_degree"
            ],
        )

        # setup the Threshold path, compute masses if necessary
        masses = None
        if new_theory["HQ"] == "MSBAR":
            masses = msbar_masses.compute(new_theory)
        tc = ThresholdsAtlas.from_dict(new_theory, masses=masses)

        # strong coupling
        sc = Couplings.from_dict(new_theory, masses=masses)
        # setup operator grid
        self.op_grid = OperatorGrid.from_dict(new_theory, new_operators, tc, sc, bfd)

        self.out = EKO.new(theory=theory_card, operator=new_operators)

    def get_output(self) -> EKO:
        """Run evolution and generate output operator.

        Two steps are applied sequentially:

        1. evolution is performed, computing the evolution operator in an
           internal flavor and x-space basis
        2. bases manipulations specified in the runcard are applied

        Returns
        -------
        EKO
            output instance

        """
        # add all operators
        for final_scale, op in self.op_grid.compute().items():
            self.out[float(final_scale)] = Operator.from_dict(op)

        return copy.deepcopy(self.out)
