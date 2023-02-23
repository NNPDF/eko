"""Define operators container and computing workflow.

The first is the driver class of eko as it is the one that collects all the
previously instantiated information and does the actual computation of the Q2s.

"""

import logging
import numbers

import numpy as np
import numpy.typing as npt

from .. import member
from .. import scale_variations as sv
from ..io.runcards import Configs, Debug
from ..thresholds import flavor_shift, is_downward_path
from . import Operator, flavors, matching_condition, physical
from .operator_matrix_element import OperatorMatrixElement

logger = logging.getLogger(__name__)


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
        mu2grid: npt.NDArray,
        order: tuple,
        masses: tuple,
        mass_scheme,
        intrinsic_flavors: list,
        xif: float,
        configs: Configs,
        debug: Debug,
        thresholds_config,
        couplings,
        interpol_dispatcher,
    ):
        """Initialize `OperatorGrid`.

        Parameters
        ----------
        config: dict
            configuration dictionary
        q2_grid: array
            Grid in Q2 on where to to compute the operators
        order: tuple(int,int)
            orders in perturbation theory
        thresholds_config: eko.thresholds.ThresholdsAtlas
            Instance of :class:`~eko.thresholds.Threshold` containing information about the
            thresholds
        couplings: eko.couplings.StrongCoupling
            Instance of :class:`~eko.couplings.StrongCoupling` able to generate a_s for
            any q
        kernel_dispatcher: eko.kernel_generation.KernelDispatcher
            Instance of the :class:`~eko.kernel_generation.KernelDispatcher` with the
            information about the kernels

        """
        # check
        config = {}
        config["order"] = order
        config["intrinsic_range"] = intrinsic_flavors
        config["xif2"] = xif**2
        config["HQ"] = mass_scheme
        config["ModSV"] = configs.scvar_method

        for i, q in enumerate("cbt"):
            config[f"m{q}"] = masses[i]
        method = config["method"] = configs.evolution_method.value
        config["backward_inversion"] = configs.inversion_method
        config["ev_op_max_order"] = configs.ev_op_max_order
        config["ev_op_iterations"] = configs.ev_op_iterations
        config["n_integration_cores"] = configs.n_integration_cores
        config["debug_skip_singlet"] = debug.skip_singlet
        config["debug_skip_non_singlet"] = debug.skip_non_singlet
        config["polarized"] = configs.polarized
        config["time_like"] = configs.time_like

        if method not in [
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
        if order == (1, 0) and method != "iterate-exact":
            logger.warning("Evolution: In LO we use the exact solution always!")

        logger.info(dict(polarized=configs.polarized))

        self.config = config
        self.q2_grid = mu2grid
        self.managers = dict(
            thresholds_config=thresholds_config,
            couplings=couplings,
            interpol_dispatcher=interpol_dispatcher,
        )
        self._threshold_operators = {}
        self._matching_operators = {}

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
            path: list(`eko.thresholds.PathSegment`)
                thresholds path

        Returns
        -------
            thr_ops: list(eko.evolution_operator.Operator)
        """
        # The base area is always that of the reference q
        thr_ops = []
        # is_downward point to smaller nf
        is_downward = is_downward_path(path)
        shift = flavor_shift(is_downward)
        for seg in path[:-1]:
            new_op_key = seg.tuple
            thr_config = self.managers["thresholds_config"]
            kthr = thr_config.thresholds_ratios[seg.nf - shift]
            ome = OperatorMatrixElement(
                self.config,
                self.managers,
                seg.nf - shift + 3,
                seg.q2_to,
                is_downward,
                np.log(kthr),
                self.config["HQ"] == "MSBAR",
            )
            if new_op_key not in self._threshold_operators:
                # Compute the operator and store it
                logger.info("Prepare threshold operator")
                op_th = Operator(
                    self.config,
                    self.managers,
                    seg.nf,
                    seg.q2_from,
                    seg.q2_to,
                    is_threshold=True,
                )
                op_th.compute()
                self._threshold_operators[new_op_key] = op_th
            thr_ops.append(self._threshold_operators[new_op_key])

            # Compute the matching conditions and store it
            if seg.q2_to not in self._matching_operators:
                ome.compute()
                self._matching_operators[seg.q2_to] = ome.op_members
        return thr_ops

    def compute(self, q2grid=None):
        """Compute all ekos for the `q2grid`.

        Parameters
        ----------
        q2grid: list(float)
            List of :math:`Q^2`

        Returns
        -------
        list(dict)
            List of ekos for each value of :math:`Q^2`
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
            # shift path for expanded scheme
            q2_gen = (
                q2 * self.config["xif2"] if self.sv_mode == sv.Modes.expanded else q2
            )
            grid_return[q2] = self.generate(q2_gen)
        return grid_return

    def generate(self, q2):
        r"""Compute a single EKO.

        Parameters
        ----------
        q2: float
            Target value of :math:`Q^2`

        Returns
        -------
        dict
            eko :math:`\mathbf E(Q^2 \leftarrow Q_0^2)` in flavor basis as numpy array
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
        is_downward = is_downward_path(path)
        if is_downward:
            intrinsic_range = [4, 5, 6]
        qed = self.config["order"][1] > 0
        final_op = physical.PhysicalOperator.ad_to_evol_map(
            operator.op_members, operator.nf, operator.q2_to, intrinsic_range, qed
        )
        # and multiply the lower ones from the right
        for op in reversed(list(thr_ops)):
            phys_op = physical.PhysicalOperator.ad_to_evol_map(
                op.op_members, op.nf, op.q2_to, intrinsic_range, qed
            )

            # join with the basis rotation, since matching requires c+ (or likewise)
            if is_downward:
                matching = matching_condition.MatchingCondition.split_ad_to_evol_map(
                    self._matching_operators[op.q2_to],
                    op.nf - 1,
                    op.q2_to,
                    intrinsic_range=intrinsic_range,
                    qed=qed,
                )
                invrot = member.ScalarOperator.promote_names(
                    flavors.rotate_matching_inverse(op.nf, qed), op.q2_to
                )
                final_op = final_op @ matching @ invrot @ phys_op
            else:
                matching = matching_condition.MatchingCondition.split_ad_to_evol_map(
                    self._matching_operators[op.q2_to],
                    op.nf,
                    op.q2_to,
                    intrinsic_range=intrinsic_range,
                    qed=qed,
                )
                rot = member.ScalarOperator.promote_names(
                    flavors.rotate_matching(op.nf + 1, qed), op.q2_to
                )
                final_op = final_op @ rot @ matching @ phys_op
        values, errors = final_op.to_flavor_basis_tensor(qed)
        return {
            "operator": values,
            "error": errors,
        }
