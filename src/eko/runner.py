# -*- coding: utf-8 -*-
"""
    This file contains the main application class of eko
"""
import logging
import copy

from eko import interpolation
from eko.kernel_generation import KernelDispatcher
from eko.thresholds import ThresholdsConfig
from eko.operator_grid import OperatorGrid
from eko.constants import Constants
from eko.strong_coupling import StrongCoupling
from eko.output import Output

logger = logging.getLogger(__name__)


class Runner:
    """
    Represents a single input configuration.

    For details about the configuration, see :doc:`here </Code/IO>`

    Parameters
    ----------
        setup : dict
            input configurations
    """

    def __init__(self, setup):
        # Print theory id setup
        logger.info("init Runner with %s", setup)
        self.out = Output()
        if setup.get("keep_input", False):
            self.out.update(setup)

        # Load constants and compute parameters
        constants = Constants()
        # setup basis grid
        bfd = interpolation.InterpolatorDispatcher.from_dict(setup)
        self.out.update(bfd.to_dict())
        # Generate the dispatcher for the kernels
        kd = KernelDispatcher.from_dict(setup, bfd, constants)
        # FNS
        tc = ThresholdsConfig.from_dict(setup)
        self.out["q2_ref"] = float(tc.q2_ref)
        # strong coupling
        sc = StrongCoupling.from_dict(setup, tc, constants)
        # setup operator grid
        self._op_grid = OperatorGrid.from_dict(
            setup,
            tc,
            sc,
            kd,
        )

    def get_operators(self):
        """ compute the actual operators """
        operators = self._op_grid.compute_q2grid()
        return operators

    def get_output(self):
        """
        Collects all data for output (to run the evolution)

        Returns
        -------
            ret : eko.output.Output
                output instance
        """
        # add all operators
        Q2grid = {}
        for op in self.get_operators():
            final_scale = op.q2_final
            Q2grid[final_scale] = op.get_raw_operators()
        self.out["Q2grid"] = Q2grid
        return copy.deepcopy(self.out)
