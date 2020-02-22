"""
    This module contains the OperatorGrid class
    q inside this class refers always to q^{2}
"""

import numpy as np
from eko.operator import Operator
import logging
logger = logging.getLogger(__name__)

class OperatorMaster:
    """
        The OperatorMaster is instantiated for a given set of parameters
        And informs the generation of operators
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
        self._integrands_ns = self._kernel_dispatcher.get_non_singlet_for_nf(self._nf)
        self._integrands_s = self._kernel_dispatcher.get_singlet_for_nf(self._nf)

    def get_op(self, q_from, q_to, generate = False):
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
    """

    def __init__(self, threshold_holder, alpha_generator, kernel_dispatcher, xgrid):
        self._threshold_holder = threshold_holder
        self._op_masters = {}
        for nf in threshold_holder.nf_range():
            # Compile the kernels for each nf
            kernel_dispatcher.set_up_all_integrands(nf)
            # Set up the OP Master for each nf
            self._op_masters[nf] = OperatorMaster(alpha_generator, kernel_dispatcher, xgrid, nf)
        self._alpha_gen = alpha_generator
        self._kernels = kernel_dispatcher
        self._threshold_operators = {}
        self._op_grid = {}
        self.qmax = -1
        self.qmin = np.inf

    def _generate_thresholds_op(self, area_list):
        """ Generate the threshold operators """
        # Get unique areas
        q_from = self._threshold_holder.qref
        nf = self._threshold_holder.nf_ref
        for area in area_list:
            q_to = area.qref
            if q_to == q_from:
                continue
            new_op = (q_from, q_to)
            if new_op not in self._threshold_operators:
                self._threshold_operators[new_op] = self._op_masters[nf].get_op(q_from, q_to, generate=True)

            nf = area.nf
            q_from = q_to

    def _get_jumps(self, q):
        """ Receives a value of q and generates a list of operators to multiply for in order to get
        down to q0 """
        full_area_path = self._threshold_holder.get_path_from_q0(q)
        # The last one is where q resides so it is not needed
        area_path = full_area_path[:-1]
        op_list = []
        for area in area_path:
            q_from = area.qref
            q_to = area.q_towards(q)
            op_list.append(self._threshold_operators[(q_from, q_to)])
        # Now get the instructions set for the composition of operators
        return op_list

    def set_q_limits(self, qmin, qmax):
        """ Sets up the limits of the grid in q^2 to be computed by the OperatorGrid

        This function computes the necessary operators to go between areas

        Parameters
        ----------
            qmin: float
                Minimum value of q that will be computed
            qmax: float
                Maximum value of q that will be computed
        """
        if qmin <= 0.0:
            raise ValueError(f"Values of q below 0.0 are not accepted, received {qmin}")
        if qmin > qmax:
            raise ValueError(f"Minimum q is above maximum q (error: {qmax} < {qmin})")
        # Get the path from q0 to qmin and qmax
        from_qmin = self._threshold_holder.get_path_from_q0(qmin)
        from_qmax = self._threshold_holder.get_path_from_q0(qmax)
        self._generate_thresholds_op(from_qmin)
        self._generate_thresholds_op(from_qmax)

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

    def get_op_at_Q(self, q):
        """
            Return the operator at Q
        """
        # Check the path to q0 for this operator
        if q in self._op_grid:
            operator = self._op_grid[q]
        else:
            self.compute_qgrid(q)
            logger.warning("Q=%f not found in the grid, computing...", q)
            operator = self._op_grid[q]
        qref = operator.qref
        nf = operator.nf
        # Prepare the path for the composition of the operator
        operators_to_q0 = self._get_jumps(qref)
        number_of_thresholds = len(operators_to_q0)
        instruction_set = self._threshold_holder.get_composition_path(nf, number_of_thresholds)
        # Compose and return
        final_op = operator.compose(operators_to_q0, instruction_set)
        return final_op.ret
