# -*- coding: utf-8 -*-
"""
This module contains the :class:`OperatorGrid` class.

The first is the driver class of eko as it is the one that collects all the
previously instantiated information and does the actual computation of the Q2s.
"""

import logging
import numbers

import numpy as np

from eko import matching_conditions, member
from eko.evolution_operator import flavors

from ..matching_conditions.operator_matrix_element import OperatorMatrixElement
from . import Operator, physical

logger = logging.getLogger(__name__)


class OperatorGrid:
    """
    The operator grid is the driver class of the evolution.

    It receives as input a threshold holder and a generator of a_s.
    From that point onwards it can compute any operator at any q2.

    Parameters
    ----------
        config: dict
            configuration dictionary
        q2_grid: array
            Grid in Q2 on where to to compute the operators
        order: int
            order in perturbation theory
        thresholds_config: eko.thresholds.ThresholdsAtlas
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
        self._threshold_operators = {}
        self._matching_operators = {}

    @classmethod
    def from_dict(
        cls,
        theory_card,
        operators_card,
        thresholds_config,
        strong_coupling,
        interpol_dispatcher,
    ):
        """
        Create the object from the theory dictionary.

        Parameters
        ----------
            theory_card : dict
                theory dictionary
            thresholds_config : eko.thresholds.ThresholdsAtlas
                An instance of the ThresholdsAtlas class
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
        config["order"] = int(theory_card["PTO"])
        method = theory_card["ModEv"]
        mod_ev2method = {
            "EXA": "iterate-exact",
            "EXP": "iterate-expanded",
            "TRN": "truncated",
        }
        method = mod_ev2method.get(method, method)
        config["method"] = method
        config["backward_inversion"] = operators_card["backward_inversion"]
        config["fact_to_ren"] = (theory_card["fact_to_ren_scale_ratio"]) ** 2
        config["ev_op_max_order"] = operators_card["ev_op_max_order"]
        config["ev_op_iterations"] = operators_card["ev_op_iterations"]
        config["debug_skip_singlet"] = operators_card["debug_skip_singlet"]
        config["debug_skip_non_singlet"] = operators_card["debug_skip_non_singlet"]
        q2_grid = np.array(operators_card["Q2grid"], np.float_)
        intrinsic_range = []
        if int(theory_card["IC"]) == 1:
            intrinsic_range.append(4)
        if int(theory_card["IB"]) == 1:
            intrinsic_range.append(5)
        config["intrinsic_range"] = intrinsic_range
        for hq in flavors.quark_names[3:]:
            config[f"m{hq}"] = theory_card[f"m{hq}"]
        return cls(
            config, q2_grid, thresholds_config, strong_coupling, interpol_dispatcher
        )

    def get_threshold_operators(self, path):
        """
        Generate the threshold operators.

        This method is called everytime the OperatorGrid is asked for a grid on Q^2
        with a list of the relevant areas.
        If new threshold operators need to be computed, they will be
        cached in an internal dictionary.

        The internal dictionary is self._threshold_operators and its structure is:
        (q2_from, q2_to) -> eko.operators.Operator

        It computes and stores the necessary macthing operators
        Parameters
        ----------
            path: list(PathSegment)
                thresholds path

        Returns
        -------
            thr_ops: list(eko.evolution_operator.Operator)
        """
        # The base area is always that of the reference q
        thr_ops = []
        for seg in path[:-1]:
            new_op_key = seg.tuple
            ome = OperatorMatrixElement(self.config, self.managers, seg.is_backward)
            if new_op_key not in self._threshold_operators:
                # Compute the operator and store it
                logger.info("Prepare threshold operator")
                op_th = Operator(
                    self.config, self.managers, seg.nf, seg.q2_from, seg.q2_to
                )
                op_th.compute()
                self._threshold_operators[new_op_key] = op_th
            thr_ops.append(self._threshold_operators[new_op_key])

            # Compute the matching conditions and store it
            if seg.q2_to not in self._matching_operators:
                # is_backawd point to the smaller q2
                shift = 3 if not seg.is_backward else 4
                kthr = self.managers["thresholds_config"].thresholds_ratios[
                    seg.nf - shift
                ]
                ome.compute(seg.q2_to, np.log(kthr))
                self._matching_operators[seg.q2_to] = ome.ome_members
        return thr_ops

    def compute(self, q2grid=None):
        """
        Computes all ekos for the q2grid.

        Parameters
        ----------
            q2grid: list(float)
                List of q^2

        Returns
        -------
            grid_return: list(dict)
                List of ekos for each value of q^2
        """
        # use input?
        if q2grid is None:
            q2grid = self.q2_grid
        # normalize input
        if isinstance(q2grid, numbers.Number):
            q2grid = [q2grid]
        # And now return the grid
        grid_return = {}
        for q2 in q2grid:
            grid_return[q2] = self.generate(q2)
        return grid_return

    def generate(self, q2):
        """
        Computes an single EKO.

        Parameters
        ----------
            q2: float
                Target value of q^2

        Returns
        -------
            final_op: dict
                eko E(q^2 <- q_0^2) in flavor basis as numpy array
        """
        # The lists of areas as produced by the thresholds
        path = self.managers["thresholds_config"].path(q2)
        # Prepare the path for the composition of the operator
        thr_ops = self.get_threshold_operators(path)
        # we start composing with the highest operator ...
        operator = Operator(
            self.config, self.managers, path[-1].nf, path[-1].q2_from, path[-1].q2_to
        )
        operator.compute()
        intrinsic_range = self.config["intrinsic_range"]
        if path[-1].is_backward:
            intrinsic_range = [4, 5, 6]
        final_op = physical.PhysicalOperator.ad_to_evol_map(
            operator.op_members,
            operator.nf,
            operator.q2_to,
            intrinsic_range,
        )

        # and multiply the lower ones from the right
        for i, op in reversed(list(enumerate(thr_ops))):
            is_backward = path[i].is_backward
            phys_op = physical.PhysicalOperator.ad_to_evol_map(
                op.op_members, op.nf, op.q2_to, intrinsic_range
            )

            # join with the basis rotation, since matching requires c+ (or likewise)
            if is_backward:
                matching = matching_conditions.MatchingCondition.split_ad_to_evol_map(
                    self._matching_operators[op.q2_to],
                    op.nf - 1,
                    op.q2_to,
                    intrinsic_range=intrinsic_range,
                )
                invrot = member.ScalarOperator.promote_names(
                    flavors.rotate_matching_inverse(op.nf), op.q2_to
                )
                final_op = final_op @ matching @ invrot @ phys_op
            else:
                matching = matching_conditions.MatchingCondition.split_ad_to_evol_map(
                    self._matching_operators[op.q2_to],
                    op.nf,
                    op.q2_to,
                    intrinsic_range=intrinsic_range,
                )
                rot = member.ScalarOperator.promote_names(
                    flavors.rotate_matching(op.nf + 1), op.q2_to
                )
                final_op = final_op @ rot @ matching @ phys_op

        values, errors = final_op.to_flavor_basis_tensor()
        return {"operators": values, "operator_errors": errors}
