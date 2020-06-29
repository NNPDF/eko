# -*- coding: utf-8 -*-
"""
    This file contains the main application class of eko
"""
import logging

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

        # Load constants and compute parameters
        self._constants = Constants()
        # setup basis grid
        self._basis_function_dispatcher = interpolation.InterpolatorDispatcher.from_dict(
            setup
        )
        # Generate the dispatcher for the kernels
        kernel_dispatcher = KernelDispatcher(
            self._basis_function_dispatcher, self._constants
        )
        # FNS
        self._threshold_holder = ThresholdsConfig.from_dict(setup)
        # strong coupling
        self._a_s = StrongCoupling.from_dict(
            setup, self._constants, self._threshold_holder
        )

        # setup operator grid
        self._op_grid = OperatorGrid(
            self._threshold_holder,
            self._a_s,
            kernel_dispatcher,
            self._basis_function_dispatcher.xgrid_raw,
        )
        self._q2grid = setup["Q2grid"]

    def get_operators(self):
        """ compute the actual operators """
        operators = self._op_grid.compute_q2grid(self._q2grid)
        return operators

    def get_output(self):
        """
            Collects all data for output (to run the evolution)

            Returns
            -------
                ret : eko.output.Output
                    output instance
        """
        # propagate grid
        ret = Output()
        ret.update(self._basis_function_dispatcher.to_dict())
        ret["q2_ref"] = float(self._threshold_holder.q2_ref)
        # add all operators
        Q2grid = {}
        for op in self.get_operators():
            final_scale = op.q2_final
            Q2grid[final_scale] = op.get_raw_operators()
        ret["Q2grid"] = Q2grid
        return ret
