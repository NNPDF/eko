"""Define operators container and computing workflow.

The first is the driver class of eko as it is the one that collects all the
previously instantiated information and does the actual computation of the Q2s.

"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from .. import member
from .. import scale_variations as sv
from ..couplings import Couplings
from ..interpolation import InterpolatorDispatcher
from ..io.runcards import Configs, Debug
from ..io.types import EvolutionPoint as EPoint
from ..io.types import Order, SquaredScale
from ..matchings import Atlas, Segment, flavor_shift, is_downward_path
from . import Operator, OpMembers, flavors, matching_condition, physical
from .operator_matrix_element import OperatorMatrixElement

logger = logging.getLogger(__name__)

OpDict = Dict[str, Optional[npt.NDArray]]
"""In particular, only the ``operator`` and ``error`` fields are expected."""


@dataclass(frozen=True)
class Managers:
    """Set of steering objects."""

    atlas: Atlas
    couplings: Couplings
    interpolator: InterpolatorDispatcher


class OperatorGrid(sv.ModeMixin):
    """Collection of evolution operators for several scales.

    The operator grid is the driver class of the evolution.

    It receives as input a threshold holder and a generator of a_s.
    From that point onwards it can compute any operator at any q2.

    Attributes
    ----------
    config: dict
    q2_grid: np.ndarray
    managers: dict

    """

    def __init__(
        self,
        mu2grid: List[EPoint],
        order: Order,
        masses: List[float],
        mass_scheme,
        thresholds_ratios: List[float],
        xif: float,
        n3lo_ad_variation: tuple,
        matching_order: Order,
        configs: Configs,
        debug: Debug,
        atlas: Atlas,
        couplings: Couplings,
        interpol_dispatcher: InterpolatorDispatcher,
        use_fhmruvv: bool,
    ):
        # check
        config: Dict[str, Any] = {}
        config["order"] = order
        config["xif2"] = xif**2
        config["HQ"] = mass_scheme
        config["ModSV"] = configs.scvar_method
        config["n3lo_ad_variation"] = n3lo_ad_variation
        config["use_fhmruvv"] = use_fhmruvv

        for i, q in enumerate("cbt"):
            config[f"m{q}"] = masses[i]
        config["thresholds_ratios"] = thresholds_ratios
        method = config["method"] = configs.evolution_method.value
        config["backward_inversion"] = configs.inversion_method
        config["ev_op_max_order"] = configs.ev_op_max_order
        config["ev_op_iterations"] = configs.ev_op_iterations
        config["n_integration_cores"] = configs.n_integration_cores
        config["debug_skip_singlet"] = debug.skip_singlet
        config["debug_skip_non_singlet"] = debug.skip_non_singlet
        config["polarized"] = configs.polarized
        config["time_like"] = configs.time_like
        config["matching_order"] = matching_order

        if order == (1, 0) and method != "iterate-exact":
            logger.warning("Evolution: In LO we use the exact solution always!")

        logger.info(dict(polarized=configs.polarized))
        logger.info(dict(time_like=configs.time_like))

        self.config = config
        self.q2_grid = mu2grid
        self.managers = Managers(
            atlas=atlas,
            couplings=couplings,
            interpolator=interpol_dispatcher,
        )
        self._threshold_operators: Dict[Segment, Operator] = {}
        self._matching_operators: Dict[SquaredScale, OpMembers] = {}

    def get_threshold_operators(self, path: List[Segment]) -> List[Operator]:
        """Generate the threshold operators.

        This method is called everytime the OperatorGrid is asked for a grid on Q^2
        with a list of the relevant areas.
        If new threshold operators need to be computed, they will be
        cached in an internal dictionary.

        The internal dictionary is self._threshold_operators and its structure is:
        (q2_from, q2_to) -> eko.operators.Operator

        It computes and stores the necessary macthing operators.

        """
        # The base area is always that of the reference q
        thr_ops = []
        # is_downward point to smaller nf
        is_downward = is_downward_path(path)
        shift = flavor_shift(is_downward)
        for seg in path[:-1]:
            kthr = self.config["thresholds_ratios"][seg.nf - shift]
            ome = OperatorMatrixElement(
                self.config,
                self.managers,
                seg.nf - shift + 3,
                seg.target,
                is_downward,
                np.log(kthr),
                self.config["HQ"] == "MSBAR",
            )
            if seg not in self._threshold_operators:
                # Compute the operator and store it
                logger.info("Prepare threshold operator")
                op_th = Operator(self.config, self.managers, seg, is_threshold=True)
                op_th.compute()
                self._threshold_operators[seg] = op_th
            thr_ops.append(self._threshold_operators[seg])

            # Compute the matching conditions and store it
            if seg.target not in self._matching_operators:
                ome.compute()
                self._matching_operators[seg.target] = ome.op_members
        return thr_ops

    def compute(self) -> Dict[EPoint, dict]:
        """Compute all ekos for the `q2grid`."""
        return {q2: self.generate(q2) for q2 in self.q2_grid}

    def generate(self, q2: EPoint) -> OpDict:
        r"""Compute a single EKO.

        eko :math:`\mathbf E(Q^2 \leftarrow Q_0^2)` in flavor basis as numpy array.

        """
        # The lists of areas as produced by the thresholds
        path = self.managers.atlas.path(q2)
        # Prepare the path for the composition of the operator
        thr_ops = self.get_threshold_operators(path)
        # we start composing with the highest operator ...
        operator = Operator(self.config, self.managers, path[-1])
        operator.compute()

        is_downward = is_downward_path(path)
        qed = self.config["order"][1] > 0

        final_op = physical.PhysicalOperator.ad_to_evol_map(
            operator.op_members, operator.nf, operator.q2_to, qed
        )
        # and multiply the lower ones from the right
        for op in reversed(list(thr_ops)):
            phys_op = physical.PhysicalOperator.ad_to_evol_map(
                op.op_members, op.nf, op.q2_to, qed
            )

            # join with the basis rotation, since matching requires c+ (or likewise)
            nf_match = op.nf - 1 if is_downward else op.nf
            matching = matching_condition.MatchingCondition.split_ad_to_evol_map(
                self._matching_operators[op.q2_to],
                nf_match,
                op.q2_to,
                qed=qed,
            )
            if is_downward:
                invrot = member.ScalarOperator.promote_names(
                    flavors.rotate_matching_inverse(op.nf, qed), op.q2_to
                )
                final_op = final_op @ matching @ invrot @ phys_op
            else:
                rot = member.ScalarOperator.promote_names(
                    flavors.rotate_matching(op.nf + 1, qed), op.q2_to
                )
                final_op = final_op @ rot @ matching @ phys_op
        values, errors = final_op.to_flavor_basis_tensor(qed)
        return {"operator": values, "error": errors}
