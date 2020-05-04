# -*- coding: utf-8 -*-
"""
    This file contains the main application class of eko
"""
import logging

import numpy as np

import eko.interpolation as interpolation
from eko.kernel_generation import KernelDispatcher
from eko.thresholds import Threshold
from eko.operator_grid import OperatorGrid
from eko.constants import Constants
from eko.alpha_s import StrongCoupling
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
        self.__init_grid(setup)
        # Generate the dispatcher for the kernels
        kernel_dispatcher = KernelDispatcher(
            self._basis_function_dispatcher, self._constants
        )
        # FNS
        self.__init_FNS(setup)

        # setup operator grid
        self._op_grid = OperatorGrid(
            self._threshold_holder,
            self._alpha_s,
            kernel_dispatcher,
            self._basis_function_dispatcher.xgrid_raw,
        )
        self._q2grid = setup["Q2grid"]

    def __init_grid(self, setup):
        """
            Setup interpolation.

            Parameters
            ----------
                setup : dict
                    input configurations
        """
        xgrid = interpolation.generate_xgrid(**setup)
        is_log_interpolation = bool(setup.get("log_interpol", True))
        polynom_rank = setup.get("xgrid_polynom_rank", 4)
        logger.info("Interpolation mode: %s", setup["xgrid_type"])
        logger.info("Log interpolation: %s", is_log_interpolation)

        # Generate the dispatcher for the basis functions
        self._basis_function_dispatcher = interpolation.InterpolatorDispatcher(
            xgrid, polynom_rank, log=is_log_interpolation
        )

    def __init_FNS(self, setup):
        """
            Get the scheme, i.e. and the thresholds and the strong coupling.

            Parameters
            ----------
                setup : dict
                    input configurations
        """
        # TODO the setup dictionary is a mess tbh
        FNS = setup["FNS"]
        q2_ref = pow(setup["Q0"], 2)
        if FNS != "FFNS":
            qmc = setup["Qmc"]
            qmb = setup["Qmb"]
            qmt = setup["Qmt"]
            threshold_list = pow(np.array([qmc, qmb, qmt]), 2)
            nf = None
        else:
            nf = setup["NfFF"]
            threshold_list = None
        self._threshold_holder = Threshold(
            q2_ref=q2_ref, scheme=FNS, threshold_list=threshold_list, nf=nf
        )

        # Now generate the operator alpha_s class
        alpha_ref = setup["alphas"]
        q2_alpha = pow(setup["Qref"], 2)
        self._alpha_s = StrongCoupling(
            self._constants, alpha_ref, q2_alpha, self._threshold_holder
        )

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
        # TODO follow yadism and create an extra Output class?
        # propagate grid
        ret = Output()
        ret.update(self._basis_function_dispatcher.get_grid_configuration())
        # add all operators
        ret["q2_ref"] = self._threshold_holder.q2_ref
        q2_grid = {}
        operators = self.get_operators()
        for op in operators:
            final_scale = op.q2_final
            q2_grid[final_scale] = op.get_operator_matrices()
        ret["q2_grid"] = q2_grid
        return ret
