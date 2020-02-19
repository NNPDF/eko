"""
    This module contains the OperatorGrid class
    q inside this class refers always to q^{2}
"""

import numpy as np
from eko.operator import Operator

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

    def __init__(self, threshold_holder, alpha_generator, kernel_dispatcher):
        self._threshold_holder = threshold_holder
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
        for area in area_list:
            q_to = area.qref
            if q_to == q_from:
                continue
            new_op = (q_from, q_to)
            if new_op not in self._threshold_operators:
                nf = area.nf
                delta_t = self._alpha_gen.delta_t(q_from, q_to)
                self._threshold_operators[new_op] = Operator(q_from, q_to, delta_t, nf, None)
            q_from = q_to

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
        # Ensure that the kernels are compiled for all possible values of nf
        nf_values = [a.nf for a in set(area_list)]
        self._kernels.set_up_all_integrands(nf_values)
        for area, q in zip(area_list, qgrid):
            q_from = area.qref
            nf = area.nf
            delta_t = self._alpha_gen.delta_t(q_from, q)
            self._op_grid[q] = Operator(q_from, q, delta_t, nf, None)

    def compute_qgrid(self, qgrid):
        """ Receives a grid in q^2 and computes all operations necessary
        to return any operator at any given q for the evolution between qref and qgrid

        Parameters
        ----------
            qgrid: list
                List of q^2
        """
        if isinstance(qgrid, (np.float, np.int)):
            qgrid = [qgrid]
        # Check max and min of the grid and reset the limits if necessary
        qmax = np.max(qgrid)
        qmin = np.min(qgrid)
        self.set_q_limits(qmin, qmax)
        # Now compute all raw operators
        self._compute_raw_grid(qgrid)


#     def get_op_at_Q(self, q):
#         """
#             Return the operator at Q
#         """
#         # Check the path to q0 for this operator
#         operator = self._op_grid.get(q)
#         if operator is None:
#             raise ValueError(f"The operator for q={q} is not registered in the grid")
#         qref = operator.qref
#         path_to_q0 = self._threshold_holder.get_path_from_q0(qref)
#         # Now get the operators from _threshold_op and multiply them
#         for th_op in path_to_q0:
#             operator = operator*th_op
#         # Now do the multiplication of these operators
#         return operator
# 
