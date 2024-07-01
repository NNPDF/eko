"""The |OME| for the non-trivial matching conditions in the |VFNS| evolution."""

import copy
import logging

import ekors
import numba as nb
import numpy as np

import ekore.operator_matrix_elements.polarized.space_like as ome_ps
import ekore.operator_matrix_elements.unpolarized.space_like as ome_us
import ekore.operator_matrix_elements.unpolarized.time_like as ome_ut

from .. import basis_rotation as br
from .. import scale_variations as sv
from ..io.types import InversionMethod
from ..matchings import Segment
from . import Operator
from .quad_ker import cb_quad_ker_ome

logger = logging.getLogger(__name__)


@nb.njit(cache=True)
def build_ome(A, matching_order, a_s, backward_method):
    r"""Construct the matching expansion in :math:`a_s` with the appropriate method.

    Parameters
    ----------
    A : numpy.ndarray
        list of |OME|
    matching_order : tuple(int,int)
        perturbation matching order
    a_s : float
        strong coupling, needed only for the exact inverse
    backward_method : InversionMethod or None
        empty or method for inverting the matching condition (exact or expanded)

    Returns
    -------
    ome : numpy.ndarray
        matching operator matrix

    """
    # to get the inverse one can use this FORM snippet
    # Symbol a;
    # NTensor c,d,e;
    # Local x=-(a*c+a**2* d + a**3 * e);
    # Local bi = 1+x+x**2+x**3;
    # Print;
    # .end
    ome = np.eye(len(A[0]), dtype=np.complex_)
    A = A[:, :, :]
    A = np.ascontiguousarray(A)
    if backward_method is InversionMethod.EXPANDED:
        # expended inverse
        if matching_order[0] >= 1:
            ome -= a_s * A[0]
        if matching_order[0] >= 2:
            ome += a_s**2 * (-A[1] + A[0] @ A[0])
        if matching_order[0] >= 3:
            ome += a_s**3 * (-A[2] + A[0] @ A[1] + A[1] @ A[0] - A[0] @ A[0] @ A[0])
    else:
        # forward or exact inverse
        if matching_order[0] >= 1:
            ome += a_s * A[0]
        if matching_order[0] >= 2:
            ome += a_s**2 * A[1]
        if matching_order[0] >= 3:
            ome += a_s**3 * A[2]
        # need inverse exact ?  so add the missing pieces
        if backward_method is InversionMethod.EXACT:
            ome = np.linalg.inv(ome)
    return ome


class OperatorMatrixElement(Operator):
    r"""
    Internal representation of a single |OME|.

    The actual matrices are computed upon calling :meth:`compute`.

    Parameters
    ----------
    config : dict
        configuration
    managers : dict
        managers
    nf: int
        number of active flavor below threshold
    q2: float
        squared matching scale
    is_backward: bool
        True for backward matching
    L: float
        :math:`\ln(\mu_F^2 / m_h^2)`
    is_msbar: bool
        add the |MSbar| contribution
    """

    log_label = "Matching"
    # complete list of possible matching operators labels
    full_labels = [
        *br.singlet_labels,
        (br.matching_hplus_pid, 21),
        (br.matching_hplus_pid, 100),
        (21, br.matching_hplus_pid),
        (100, br.matching_hplus_pid),
        (br.matching_hplus_pid, br.matching_hplus_pid),
        (200, 200),
        (200, br.matching_hminus_pid),
        (br.matching_hminus_pid, 200),
        (br.matching_hminus_pid, br.matching_hminus_pid),
    ]
    # still valid in QED since Sdelta and Vdelta matchings are diagonal
    full_labels_qed = copy.deepcopy(full_labels)

    def __init__(self, config, managers, nf, q2, is_backward, L, is_msbar):
        super().__init__(config, managers, Segment(q2, q2, nf))
        self.backward_method = config["backward_inversion"] if is_backward else None
        if is_backward:
            self.is_intrinsic = True
        else:
            self.is_intrinsic = bool(len(config["intrinsic_range"]) != 0)
        self.L = L
        self.is_msbar = is_msbar
        # Note for the moment only QCD matching is implemented
        self.order = tuple(config["matching_order"])

    @property
    def labels(self):
        """Necessary sector labels to compute.

        Returns
        -------
        list(str)
            sector labels
        """
        labels = []
        # non-singlet labels
        if self.config["debug_skip_non_singlet"]:
            logger.warning("%s: skipping non-singlet sector", self.log_label)
        else:
            labels.append((200, 200))
            if self.is_intrinsic or self.backward_method is not None:
                # intrinsic labels, which are not zero at NLO
                labels.append((br.matching_hminus_pid, br.matching_hminus_pid))
                # These contributions are always 0 for the moment
                # labels.extend([(200, br.matching_hminus_pid), (br.matching_hminus_pid, 200)])
        # same for singlet
        if self.config["debug_skip_singlet"]:
            logger.warning("%s: skipping singlet sector", self.log_label)
        else:
            labels.extend(
                [
                    *br.singlet_labels,
                    (br.matching_hplus_pid, 21),
                    (br.matching_hplus_pid, 100),
                ]
            )
            if self.is_intrinsic or self.backward_method is not None:
                labels.extend(
                    [
                        (21, br.matching_hplus_pid),
                        (100, br.matching_hplus_pid),
                        (br.matching_hplus_pid, br.matching_hplus_pid),
                    ]
                )
        return labels

    def run_op_integration(self, log_grid):
        """Run the integration for each grid point.

        Parameters
        ----------
        log_grid : tuple(k, logx)
            log grid point with relative index

        Returns
        -------
        list
            computed operators at the give grid point
        """
        column = []
        k, logx = log_grid
        # call(!) self.labels only once
        labels = self.labels
        start_time = time.perf_counter()
        # start preparing C arguments
        cfg = ekors.lib.empty_qcd_args()
        cfg.order_qcd = self.order[0]
        cfg.is_polarized = self.config["polarized"]
        cfg.is_time_like = self.config["time_like"]
        cfg.nf = self.nf
        cfg.py = ekors.ffi.cast("void *", cb_quad_ker_ome.address)
        cfg.is_log = self.int_disp.log
        cfg.logx = logx
        cfg.L = self.L
        # cfg.method_num = 1
        cfg.as1 = self.as_list[1]
        cfg.as0 = self.as_list[0]
        # cfg.ev_op_iterations = self.config["ev_op_iterations"]
        # cfg.ev_op_max_order_qcd = self.config["ev_op_max_order"][0]
        # cfg.sv_mode_num = 1
        # cfg.is_threshold = self.is_threshold
        cfg.Lsv = np.log(self.xif2)

        # iterate basis functions
        for l, bf in enumerate(self.int_disp):
            if k == l and l == self.grid_size - 1:
                continue
            # add emtpy labels with 0s
            if bf.is_below_x(np.exp(logx)):
                column.append({label: (0.0, 0.0) for label in labels})
                continue
            temp_dict = {}
            # prepare areas for C
            curareas = bf.areas_representation
            areas_len = curareas.shape[0] * curareas.shape[1]
            # force the variable in scope
            areas_ffi = ekors.ffi.new(
                f"double[{areas_len}]", curareas.flatten().tolist()
            )
            cfg.areas = areas_ffi
            cfg.areas_x = curareas.shape[0]
            cfg.areas_y = curareas.shape[1]
            # iterate sectors
            for label in labels:
                cfg.mode0 = label[0]
                cfg.mode1 = label[1]
                # construct the low level object
                func = LowLevelCallable(
                    ekors.lib.rust_quad_ker_ome, ekors.ffi.addressof(cfg)
                )
                res = integrate.quad(
                    func,
                    0.5,
                    1.0 - self._mellin_cut,
                    epsabs=1e-12,
                    epsrel=1e-5,
                    limit=100,
                    full_output=1,
                )
                temp_dict[label] = res[:2]
            column.append(temp_dict)
        logger.info(
            "%s: computing operators - %u/%u took: %6f s",
            self.log_label,
            k + 1,
            self.grid_size,
            (time.perf_counter() - start_time),
        )
        return column

    @property
    def a_s(self):
        """Return the computed values for :math:`a_s`.

        Note that here you need to use :math:`a_s^{n_f+1}`
        """
        sc = self.managers["couplings"]
        return sc.a_s(
            self.q2_from
            * (self.xif2 if self.sv_mode == sv.Modes.exponentiated else 1.0),
            nf_to=self.nf + 1,
        )

    def compute(self):
        """Compute the actual operators (i.e. run the integrations)."""
        self.initialize_op_members()

        # At LO you don't need anything else
        if self.order[0] == 0:
            logger.info("%s: no need to compute matching at LO", self.log_label)
            return
        logger.info(
            "%s: order: (%d, %d), backward method: %s",
            self.log_label,
            self.order[0],
            self.order[1],
            self.backward_method,
        )

        self.integrate()
