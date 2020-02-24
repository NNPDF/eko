"""
    This module contains the OperatorGrid class
    This is the driver class of eko as it is the one that collects all the
    previously instantiated information and does the actual computation of the Qs

    This class, however, has no knowledge about _what_ it does compute as
    everything must always come in externally

    q inside this class refers always to q^{2}
"""

# TODO:
# 1 - the operator grid should have a "save" and "load" method which will
# allow to store the full grid

import numpy as np
from eko.operator import Operator
import logging
logger = logging.getLogger(__name__)

class OperatorMaster:
    """
        The OperatorMaster is instantiated for a given set of parameters
        And informs the generation of operators.

        This class is just a convenience wrapper

        Parameters
        ----------
            alpha_generator: eko.alpha_s.StrongCoupling
                Instance of the StrongCoupling class able to generate a_s for any q
            kernel_dispatcher: eko.kernels.KernelDispatcher
                Instance of the KernelDispatcher with the information about the kernels
            xgrid: np.array
                Grid in x used to compute the operators
            nf: int
                Value of nf for this OperatorMaster
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
        self._integrands_ns = self._kernel_dispatcher.get_non_singlet_for_nf(self._nf)
        self._integrands_s = self._kernel_dispatcher.get_singlet_for_nf(self._nf)

    def get_op(self, q_from, q_to, generate = False):
        """ Given a q_from and a q_to, returns a raw operator.
        If the `generate` flag is set to True, the operator will also be computed
        in place.

        Note: this method is just for convenience and so this method cannot check
        that no thresholds are crossed in order to go from q_from to q_to.
        This is the responsability of the calling function.
        

        Parameters
        ----------
            q_from: float
                Reference value of q^2
            q_to: float
                Target value of q^2
            generate: bool
                Whether the operator should be computed (default = False)

        Returns
        -------
            op: eko.operators.Operator
                operator to go from q_from to q_to
        """
        if self._integrands_s is None or self._integrands_ns is None:
            self._compile()
        # Generate the metadata for this operator
        metadata = {
                'q' : q_to,
                'qref' : q_from,
                'nf' : self._nf
                }
        # Generate the necessary parameters to compute the operator
        delta_t = self._alpha_gen.delta_t(q_from, q_to)
        op = Operator(delta_t, self._xgrid, self._integrands_ns, self._integrands_s, metadata)
        if generate:
            op.compute()
        return op


class OperatorGrid:
    """
        The operator grid is the driver class of the evolution.
        It receives as input a threshold holder and a generator of alpha_s

        From that point onwards it can compute any operator at any q

        Parameters
        ----------
            threshold_holder: eko.thresholds.Threshold
                Instance of the Threshold class containing information about the thresholds
            alpha_generator: eko.alpha_s.StrongCoupling
                Instance of the StrongCoupling class able to generate a_s for any q
            kernel_dispatcher: eko.kernels.KernelDispatcher
                Instance of the KernelDispatcher with the information about the kernels
            xgrid: np.array
                Grid in x used to compute the operators
    """

    def __init__(self, threshold_holder, alpha_generator, kernel_dispatcher, xgrid):
        logger.info("Instantiating an operator grid:")
        logger.info(" > Flavour scheme: %s", threshold_holder.scheme)
        self._threshold_holder = threshold_holder
        logger.info(" > Reference alpha_s(Q^2=%f)=%f", alpha_generator.qref, alpha_generator.ref)
        self._alpha_gen = alpha_generator
        self._kernels = kernel_dispatcher
        # Prepare the OperatorMasters for each accepted value of nf
        self._op_masters = {}
        for nf in threshold_holder.nf_range():
            # Compile the kernels for each nf
            kernel_dispatcher.set_up_all_integrands(nf)
            # Set up the OP Master for each nf
            self._op_masters[nf] = OperatorMaster(alpha_generator, kernel_dispatcher, xgrid, nf)
        min_nf = threshold_holder.min_nf
        max_nf = threshold_holder.max_nf
        logger.info(" > Accepted nf range: [%d, %d]", min_nf, max_nf)
        self._op_grid = {}
        self._threshold_operators = {}

    def _generate_thresholds_op(self, to_q2):
        """ Generate the thresholds operators
        This class is called everythime the OperatorGrid is asked for a grid on Q^2
        with a list of the relevant areas.
        If new threshold operators need to be computed, they will and they will be
        cached in an internal dictionary.

        The internal dictionary is self._threshold_operators and its structure is:
            (q_from, q_to) : eko.operators.Operator

        Parameters
        ----------
            to_q2: float
                value of q for which the OperatorGrid will need to pass thresholds
        """
        # The lists of areas as produced by the self._threshold_holder
        area_list = self._threshold_holder.get_path_from_q0(to_q2)
        # The base area is always that of the reference q
        q_from = self._threshold_holder.qref
        nf = self._threshold_holder.nf_ref
        for area in area_list:
            q_to = area.qref
            if q_to == q_from:
                continue
            new_op = (q_from, q_to)
            if new_op not in self._threshold_operators:
                # Compute the operator in place and store it
                op_th = self._op_masters[nf].get_op(q_from, q_to, generate=True)
                self._threshold_operators[new_op] = op_th
            nf = area.nf
            q_from = q_to

    def _get_jumps(self, qsq):
        """ Given a value of q^2, generates the list of operators that need to be
        composed in order to get there from q0^2

        Parameters
        ----------
            qsq: float
                Target value of q^2

        Returns
        -------
            op_list: list
                List of threhsold operators
        """
        # Get the list of areas to be crossed
        full_area_path = self._threshold_holder.get_path_from_q0(qsq)
        # The last one is where q resides so it is not needed
        area_path = full_area_path[:-1]
        op_list = []
        # Now loop over the areas to collect the necessary threshold operators
        for area in area_path:
            q_from = area.qref
            q_to = area.q_towards(qsq)
            op_list.append(self._threshold_operators[(q_from, q_to)])
        return op_list

    def set_q_limits(self, qmin, qmax):
        """ Sets up the limits of the grid in q^2 to be computed by the OperatorGrid

        This function is a wrapper to compute the necessary operators to go between areas

        Parameters
        ----------
            qmin: float
                Minimum value of q that will be computed
            qmax: float
                Maximum value of q that will be computed
        """
        # Sanity checks
        if qmin <= 0.0 or qmax <= 0.0:
            raise ValueError(f"Values of q below 0.0 are not accepted, received [{qmin},{qmax}")
        if qmin > qmax:
            raise ValueError(f"Minimum q is above maximum q (error: {qmax} < {qmin})")
        # Ensure we have all the necessary operators to go from qref to qmin and qmax
        self._generate_thresholds_op(qmin)
        self._generate_thresholds_op(qmax)

    def _compute_raw_grid(self, qgrid):
        """ Receives a grid in q^2 and computes each opeator inside its
        area with reference value the q_ref of its area

        Parameters
        ----------
            qgrid: list
                List of q^2
        """
        area_list = self._threshold_holder.get_areas(qgrid)
        for area, q in zip(area_list, qgrid):
            q_from = area.qref
            nf = area.nf
            self._op_grid[q] = self._op_masters[nf].get_op(q_from, q)
        # Now perform the computation, TODO everything in parallel
        for _, op in self._op_grid.items():
            op.compute()

    def compute_qgrid(self, qgrid):
        """ Receives a grid in q^2 and computes all operations necessary
        to return any operator at any given q for the evolution between qref and qgrid

        Parameters
        ----------
            qgrid: list
                List of q^2

        Returns
        -------
            grid_return: list
                List of PhysicalOperator for each value of q^2
        """
        if isinstance(qgrid, (np.float, np.int, np.integer)):
            qgrid = [qgrid]
        # Check max and min of the grid and reset the limits if necessary
        qmax = np.max(qgrid)
        qmin = np.min(qgrid)
        self.set_q_limits(qmin, qmax)
        # Now compute all raw operators
        self._compute_raw_grid(qgrid)
        # And now return the grid
        grid_return = []
        for q in qgrid:
            grid_return.append(self.get_op_at_Q(q))
        return grid_return

    def get_op_at_Q(self, qsq):
        """ Given a value of q^2, returns the PhysicalOperator to get
        to q^2 from q0^2

        Parameters
        ----------
            qsq: float
                Target value of q^2

        Returns
        -------
            final_op: eko.operators.PhysicalOperator
                Op(q0^2 -> q^2)
        """
        # Check the path to q0 for this operator
        if qsq in self._op_grid:
            operator = self._op_grid[qsq]
        else:
            logger.warning("Q=%f not found in the grid, computing...", qsq)
            self.compute_qgrid(qsq)
            operator = self._op_grid[qsq]
        qref = operator.qref
        nf = operator.nf
        # Prepare the path for the composition of the operator
        operators_to_q0 = self._get_jumps(qref)
        number_of_thresholds = len(operators_to_q0)
        instruction_set = self._threshold_holder.get_composition_path(nf, number_of_thresholds)
        # Compose and return
        final_op = operator.compose(operators_to_q0, instruction_set)
        return final_op
