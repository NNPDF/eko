# -*- coding: utf-8 -*-
"""
    This module contains the :class:`OperatorGrid`  and the
    :class:`OperatorMaster` class.

    The first is the driver class of eko as it is the one that collects all the
    previously instantiated information and does the actual computation of the Qs

    The classes, however, have no knowledge about _what_ it does compute as
    everything must always come in externally.

    See :doc:`Operator overview </Code/Operators>`.
"""

import logging
import numpy as np
from eko.evolution_operator import Operator

logger = logging.getLogger(__name__)


class OperatorMaster:
    """
        The :class:`OperatorMaster` is instantiated for a given set of parameters
        And informs the generation of operators.

        This class is just a convenience wrapper.

        Parameters
        ----------
            alpha_generator: eko.strong_coupling.StrongCoupling
                Instance of the :class:`~eko.strong_coupling.StrongCoupling` class able to
                generate a_s for any q
            kernel_dispatcher: eko.kernels.KernelDispatcher
                Instance of the :class:`~eko.kernels.KernelDispatcher` with the information
                about the kernels
            xgrid: np.array
                Grid in x used to compute the operators
            nf: int
                Value of nf for this :class:`OperatorMaster`
    """

    def __init__(self, alpha_generator, kernel_dispatcher, xgrid, nf):
        # Get all the integrands necessary for singlet and not singlet for nf
        self._kernel_dispatcher = kernel_dispatcher
        self._alpha_gen = alpha_generator
        self._xgrid = xgrid
        self._nf = nf
        self._integrands_ns = None
        self._integrands_s = None

    def _compile(self):
        """ Compiles the kernels and make them become integrands """
        self._integrands_ns = self._kernel_dispatcher.integrands_ns[self._nf]
        self._integrands_s = self._kernel_dispatcher.integrands_s[self._nf]

    def get_op(self, q2_from, q2_to, generate=False):
        """
            Given a q2_from and a q2_to, returns a raw operator.
            If the `generate` flag is set to True, the operator will also be computed
            in place.

            Note: this method is just for convenience and so this method cannot check
            that no thresholds are crossed in order to go from q2_from to q2_to.
            This is the responsability of the calling function.


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
        if self._integrands_s is None or self._integrands_ns is None:
            self._compile()
        # Generate the metadata for this operator
        metadata = {"q2": q2_to, "q2ref": q2_from, "nf": self._nf}
        # Generate the necessary parameters to compute the operator
        delta_t = self._alpha_gen.delta_t(q2_from, q2_to)
        op = Operator(
            delta_t, self._xgrid, self._integrands_ns, self._integrands_s, metadata
        )
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
            threshold_holder: eko.thresholds.Threshold
                Instance of :class:`~eko.thresholds.Threshold` containing information about the
                thresholds
            alpha_generator: eko.strong_coupling.StrongCoupling
                Instance of :class:`~eko.strong_coupling.StrongCoupling` able to generate a_s for
                any q
            kernel_dispatcher: eko.kernel_generation.KernelDispatcher
                Instance of the :class:`~eko.kernel_generation.KernelDispatcher` with the
                information about the kernels
            xgrid: np.array
                Grid in x used to compute the operators
    """

    def __init__(self, threshold_holder, alpha_generator, kernel_dispatcher, xgrid):
        logger.info("Instantiating an operator grid:")
        logger.info("Flavour scheme: %s", threshold_holder.scheme)
        self._threshold_holder = threshold_holder
        logger.info(
            "Reference a_s(Q^2=%f)=%f", alpha_generator.q2_ref, alpha_generator.as_ref
        )
        self._alpha_gen = alpha_generator
        self._kernels = kernel_dispatcher
        # Prepare the OperatorMasters for each accepted value of nf
        self._op_masters = {}
        for nf in threshold_holder.nf_range():
            # Compile the kernels for each nf
            kernel_dispatcher.set_up_all_integrands(nf)
            # Set up the OperatorMaster for each nf
            self._op_masters[nf] = OperatorMaster(
                alpha_generator, kernel_dispatcher, xgrid, nf
            )
        min_nf = threshold_holder.min_nf
        max_nf = threshold_holder.max_nf
        logger.info("Accepted nf range: [%d, %d]", min_nf, max_nf)
        self._op_grid = {}
        self._threshold_operators = {}

    def _generate_thresholds_op(self, to_q2):
        """
            Generate the thresholds operators

            This method is called everytime the OperatorGrid is asked for a grid on Q^2
            with a list of the relevant areas.
            If new threshold operators need to be computed, they will and they will be
            cached in an internal dictionary.

            The internal dictionary is self._threshold_operators and its structure is:
                (q2_from, q2_to) : eko.operators.Operator

            Parameters
            ----------
                to_q2: float
                    value of q2 for which the OperatorGrid will need to pass thresholds
        """
        # The lists of areas as produced by the self._threshold_holder
        area_list = self._threshold_holder.get_path_from_q2_ref(to_q2)
        # The base area is always that of the reference q
        q2_from = self._threshold_holder.q2_ref
        nf = self._threshold_holder.nf_ref
        for area in area_list:
            q2_to = area.q2_ref
            if q2_to == q2_from:
                continue
            new_op = (q2_from, q2_to)
            logger.info("Compute threshold operator from %e to %e", q2_from, q2_to)
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
        full_area_path = self._threshold_holder.get_path_from_q2_ref(qsq)
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
        area_list = self._threshold_holder.get_areas(q2grid)
        for area, q2 in zip(area_list, q2grid):
            q2_from = area.q2_ref
            nf = area.nf
            self._op_grid[q2] = self._op_masters[nf].get_op(q2_from, q2)
        # Now perform the computation
        # TODO everything in parallel
        for _, op in self._op_grid.items():
            op.compute()

    def compute_q2grid(self, q2grid):
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
        if isinstance(q2grid, (np.float, np.int, np.integer)):
            q2grid = [q2grid]
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
                final_op: eko.operators.PhysicalOperator
                    Op(q_0^2 -> q^2)
        """
        # Check the path to q0 for this operator
        if qsq in self._op_grid:
            operator = self._op_grid[qsq]
        else:
            logger.warning("Q2=%f not found in the grid, computing...", qsq)
            self.compute_q2grid(qsq)
            operator = self._op_grid[qsq]
        q2ref = operator.q2ref
        nf = operator.nf
        # Prepare the path for the composition of the operator
        operators_to_q2 = self._get_jumps(q2ref)
        number_of_thresholds = len(operators_to_q2)
        instruction_set = self._threshold_holder.get_composition_path(
            nf, number_of_thresholds
        )
        # Compose and return
        final_op = operator.compose(operators_to_q2, instruction_set, qsq)
        return final_op
