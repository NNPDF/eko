# -*- coding: utf-8 -*-
"""
    This file contains the main application class of eko
"""
import logging
import copy

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
        # defer numba compiling inside interpolation
        from . import interpolation # pylint: disable=import-outside-toplevel
        from .output import Output # pylint: disable=import-outside-toplevel
        from .strong_coupling import StrongCoupling # pylint: disable=import-outside-toplevel
        from .thresholds import ThresholdsConfig # pylint: disable=import-outside-toplevel
        from .operator.grid import OperatorGrid # pylint: disable=import-outside-toplevel
        self.out = Output()
        if setup.get("keep_input", False):
            self.out.update(copy.deepcopy(setup))

        # setup basis grid
        bfd = interpolation.InterpolatorDispatcher.from_dict(setup)
        self.out.update(bfd.to_dict())
        # FNS
        tc = ThresholdsConfig.from_dict(setup)
        self.out["q2_ref"] = float(tc.q2_ref)
        # strong coupling
        sc = StrongCoupling.from_dict(setup, tc)
        # setup operator grid
        self._op_grid = OperatorGrid.from_dict(
            setup,
            tc,
            sc,
            bfd,
        )

    def get_operators(self):
        """compute the actual operators"""
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
            Q2grid[float(final_scale)] = op.get_raw_operators()
        self.out["Q2grid"] = Q2grid
        return copy.deepcopy(self.out)
