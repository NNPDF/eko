# -*- coding: utf-8 -*-
"""
This module contains the :class:`OperatorGrid`  and the
:class:`OperatorMaster` class.

The first is the driver class of eko as it is the one that collects all the
previously instantiated information and does the actual computation of the Q2s.
"""

import logging
import numbers
import time

import numpy as np

logger = logging.getLogger(__name__)


class OperatorMaster:
    """
    The :class:`OperatorMaster` is instantiated for a given set of parameters
    and informs the generation of operators.

    Parameters
    ----------
        grid: OperatorGrid
            Instance of :class:`OperatorGrid`
        nf: int
            Value of nf for this :class:`OperatorMaster`
    """

    def __init__(self, config, managers, nf):
        self.config = config
        self.managers = managers
        self.nf = nf

    def get_op(self, q2_from, q2_to, generate=False):
        """
        Given a q2_from and a q2_to, returns a raw operator.
        If the `generate` flag is set to True, the operator will also be computed
        in place.

        Parameters
        ----------
            q2_from: float
                Reference value of q^2
            q2_to: float
                Target value of q^2
            generate: bool
                Whether the operator should be computed (default = False)

        Returns
        -------
            op: eko.operators.Operator
                operator to go from q2_from to q2_to
        """
        # import numba late - that is only here
        start = time.perf_counter()
        logger.info("Evolution: compiling Numba")
        from .operator import Operator  # pylint: disable=import-outside-toplevel

        logger.info(
            "Evolution: compiling Numba - took %.1f s", time.perf_counter() - start
        )
        # now do the computation
        op = Operator(self.config, self.managers, self.nf, q2_from, q2_to)
        if generate:
            op.compute()
        return op


class OperatorGrid:
    """
    The operator grid is the driver class of the evolution.

    It receives as input a threshold holder and a generator of a_s.
    From that point onwards it can compute any operator at any q2.

    Parameters
    ----------
        q2_grid: array
            Grid in Q2 on where to to compute the operators
        order: int
            order in perturbation theory
        thresholds_config: eko.thresholds.ThresholdsConfig
            Instance of :class:`~eko.thresholds.Threshold` containing information about the
            thresholds
        strong_coupling: eko.strong_coupling.StrongCoupling
            Instance of :class:`~eko.strong_coupling.StrongCoupling` able to generate a_s for
            any q
        kernel_dispatcher: eko.kernel_generation.KernelDispatcher
            Instance of the :class:`~eko.kernel_generation.KernelDispatcher` with the
            information about the kernels
    """

    def __init__(
        self,
        config,
        q2_grid,
        thresholds_config,
        strong_coupling,
        interpol_dispatcher,
    ):
        # check
        order = int(config["order"])
        method = config["method"]
        if not method in [
            "iterate-exact",
            "iterate-expanded",
            "truncated",
            "ordered-truncated",
            "decompose-exact",
            "decompose-expanded",
            "perturbative-exact",
            "perturbative-expanded",
        ]:
            raise ValueError(f"Unknown evolution mode {method}")
        if order == 0 and method != "iterate-exact":
            logger.warning("Evolution: In LO we use the exact solution always!")
        self.config = config
        self.q2_grid = q2_grid
        self.managers = dict(
            thresholds_config=thresholds_config,
            strong_coupling=strong_coupling,
            interpol_dispatcher=interpol_dispatcher,
        )
        self._op_masters = {}
        self._op_grid = {}
        self._threshold_operators = {}

    @classmethod
    def from_dict(
        cls,
        setup,
        thresholds_config,
        strong_coupling,
        interpol_dispatcher,
    ):
        """
        Create the object from the theory dictionary.

        Read keys:

            - Q2grid : required, target grid
            - debug_skip_singlet : optional, avoid singlet integrations?
            - debug_skip_non_singlet : optional, avoid non-singlet integrations?

        Parameters
        ----------
            setup : dict
                theory dictionary
            thresholds_config : eko.thresholds.ThresholdsConfig
                An instance of the ThresholdsConfig class
            strong_coupling : eko.strong_coupling.StrongCoupling
                An instance of the StrongCoupling class
            interpol_dispatcher : eko.interpolation.InterpolatorDispatcher
                An instance of the InterpolatorDispatcher class

        Returns
        -------
            obj : cls
                created object
        """
        config = {}
        config["order"] = int(setup["PTO"])
        method = setup.get("ModEv", "iterate-exact")
        mod_ev2method = {
            "EXA": "iterate-exact",
            "EXP": "iterate-expanded",
            "TRN": "truncated",
        }
        method = mod_ev2method.get(method, method)
        config["method"] = method
        config["log_fact_to_ren"] = np.log((setup["XIF"] / setup["XIR"])**2)
        config["ev_op_max_order"] = setup.get("ev_op_max_order", 10)
        config["ev_op_iterations"] = setup.get("ev_op_iterations", 10)
        config["debug_skip_singlet"] = setup.get("debug_skip_singlet", False)
        config["debug_skip_non_singlet"] = setup.get("debug_skip_non_singlet", False)
        q2_grid = np.array(setup["Q2grid"], np.float_)
        return cls(
            config, q2_grid, thresholds_config, strong_coupling, interpol_dispatcher
        )

    def _generate_masters(self):
        """
        Creates all :class:`OperatorMaster` instances.
        """
        for nf in self.managers["thresholds_config"].nf_range():
            # Set up the OperatorMaster for each nf
            self._op_masters[nf] = OperatorMaster(
                self.config,
                {k: v for k, v in self.managers.items() if k != "thresholds_config"},
                nf,
            )

    def _generate_thresholds_op(self, to_q2):
        """
        Generate the threshold operators

        This method is called everytime the OperatorGrid is asked for a grid on Q^2
        with a list of the relevant areas.
        If new threshold operators need to be computed, they will be
        cached in an internal dictionary.

        The internal dictionary is self._threshold_operators and its structure is:
        (q2_from, q2_to) -> eko.operators.Operator

        Parameters
        ----------
            to_q2: float
                value of q2 for which the OperatorGrid will need to pass thresholds
        """
        # The lists of areas as produced by the thresholds
        area_list = self.managers["thresholds_config"].get_path_from_q2_ref(to_q2)
        # The base area is always that of the reference q
        q2_from = self.managers["thresholds_config"].q2_ref
        nf = self.managers["thresholds_config"].nf_ref
        for area in area_list:
            q2_to = area.q2_ref
            if q2_to == q2_from:
                continue
            new_op = (q2_from, q2_to)
            logger.info(
                "Evolution: Compute threshold operator from %e to %e", q2_from, q2_to
            )
            if new_op not in self._threshold_operators:
                # Compute the operator in place and store it
                op_th = self._op_masters[nf].get_op(q2_from, q2_to, generate=True)
                self._threshold_operators[new_op] = op_th
            nf = area.nf
            q2_from = q2_to

    def _get_jumps(self, qsq):
        """
        Given a value of q^2, generates the list of operators that need to be
        composed in order to get there from q0^2

        Parameters
        ----------
            qsq: float
                Target value of q^2

        Returns
        -------
            op_list: list
                List of threshold operators
        """
        # Get the list of areas to be crossed
        full_area_path = self.managers["thresholds_config"].get_path_from_q2_ref(qsq)
        # The last one is where q resides so it is not needed
        area_path = full_area_path[:-1]
        op_list = []
        # Now loop over the areas to collect the necessary threshold operators
        for area in area_path:
            q2_from = area.q2_ref
            q2_to = area.q2_towards(qsq)
            op_list.append(self._threshold_operators[(q2_from, q2_to)])
        return op_list

    def set_q2_limits(self, q2min, q2max):
        """
        Sets up the limits of the grid in q^2 to be computed by the OperatorGrid

        This function is a wrapper to compute the necessary operators to go between areas

        Parameters
        ----------
            q2min: float
                Minimum value of q^2 that will be computed
            q2max: float
                Maximum value of q^2 that will be computed
        """
        # Sanity checks
        if q2min <= 0.0 or q2max <= 0.0:
            raise ValueError(
                f"Values of q^2 below 0.0 are not accepted, received [{q2min},{q2max}]"
            )
        if q2min > q2max:
            raise ValueError(
                f"Minimum q^2 is above maximum q^2 (error: {q2max} < {q2min})"
            )
        # Ensure we have all the necessary operators to go from q2ref to q2min and qmax
        self._generate_thresholds_op(q2min)
        self._generate_thresholds_op(q2max)

    def _compute_raw_grid(self, q2grid):
        """
        Receives a grid in q^2 and computes each opeator inside its
        area with reference value the q_ref of its area

        Parameters
        ----------
            q2grid: list
                List of q^2
        """
        area_list = self.managers["thresholds_config"].get_areas(q2grid)
        for area, q2 in zip(area_list, q2grid):
            q2_from = area.q2_ref
            nf = area.nf
            logger.info("Evolution: compute operators %e -> %e", q2_from, q2)
            self._op_grid[q2] = self._op_masters[nf].get_op(q2_from, q2)
        # Now perform the computation
        for op in self._op_grid.values():
            op.compute()

    def compute_q2grid(self, q2grid=None):
        """
        Receives a grid in q^2 and computes all operations necessary
        to return any operator at any given q for the evolution between q2ref and q2grid

        Parameters
        ----------
            q2grid: list
                List of q^2

        Returns
        -------
            grid_return: list
                List of PhysicalOperator for each value of q^2
        """
        # use input?
        if q2grid is None:
            q2grid = self.q2_grid
        # normalize input
        if isinstance(q2grid, numbers.Number):
            q2grid = [q2grid]
        # build masters
        if len(self._op_masters) <= 0:
            self._generate_masters()
        # Check max and min of the grid and reset the limits if necessary
        q2max = np.max(q2grid)
        q2min = np.min(q2grid)
        self.set_q2_limits(q2min, q2max)
        # Now compute all raw operators
        self._compute_raw_grid(q2grid)
        # And now return the grid
        grid_return = []
        for q2 in q2grid:
            grid_return.append(self.get_op_at_q2(q2))
        return grid_return

    def get_op_at_q2(self, qsq):
        """
        Given a value of q^2, returns the PhysicalOperator to get
        to q^2 from q0^2

        Parameters
        ----------
            qsq: float
                Target value of q^2

        Returns
        -------
            final_op: eko.evolution_operator.PhysicalOperator
                Op(q^2 <- q_0^2)
        """
        # Check the path to q0 for this operator
        if qsq in self._op_grid:
            operator = self._op_grid[qsq]
        else:
            logger.warning("Evolution: Q2=%e not found in the grid, computing...", qsq)
            self.compute_q2grid(qsq)
            operator = self._op_grid[qsq]
        # Prepare the path for the composition of the operator
        operators_to_q2 = self._get_jumps(operator.q2_from)
        number_of_thresholds = len(operators_to_q2)
        instruction_set = self.managers["thresholds_config"].get_composition_path(
            operator.nf, number_of_thresholds
        )
        # Compose and return
        final_op = operator.compose(operators_to_q2, instruction_set, qsq)
        return final_op
