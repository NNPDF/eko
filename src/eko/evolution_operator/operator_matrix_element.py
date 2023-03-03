"""The |OME| for the non-trivial matching conditions in the |VFNS| evolution."""

import copy
import functools
import logging

import numba as nb
import numpy as np

import ekore.operator_matrix_elements.polarized.space_like as ome_ps
import ekore.operator_matrix_elements.unpolarized.space_like as ome_us
import ekore.operator_matrix_elements.unpolarized.time_like as ome_ut
from ekore import harmonics

from .. import basis_rotation as br
from . import Operator, QuadKerBase

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
    backward_method : ["exact", "expanded" or ""]
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
    A = np.ascontiguousarray(A)
    if backward_method == "expanded":
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
        if backward_method == "exact":
            ome = np.linalg.inv(ome)
    return ome


@nb.njit(cache=True)
def quad_ker(
    u,
    order,
    mode0,
    mode1,
    is_log,
    logx,
    areas,
    a_s,
    nf,
    L,
    backward_method,
    is_msbar,
    is_polarized,
    is_time_like,
):
    r"""Raw kernel inside quad.

    Parameters
    ----------
    u : float
        quad argument
    order : tuple(int,int)
        perturbation matching order
    mode0 : int
        pid for first element in the singlet sector
    mode1 : int
        pid for second element in the singlet sector
    is_log : boolean
        logarithmic interpolation
    logx : float
        Mellin inversion point
    areas : tuple
        basis function configuration
    a_s : float
        strong coupling, needed only for the exact inverse
    nf: int
        number of active flavor below threshold
    L : float
        :math:``\ln(\mu_F^2 / m_h^2)``
    backward_method : ["exact", "expanded" or ""]
        empty or method for inverting the matching condition (exact or expanded)
    is_msbar: bool
        add the |MSbar| contribution
    is_polarized : boolean
        is polarized evolution ?
    is_time_like : boolean
        is time-like evolution ?

    Returns
    -------
    ker : float
        evaluated integration kernel

    """
    ker_base = QuadKerBase(u, is_log, logx, mode0)
    integrand = ker_base.integrand(areas)
    if integrand == 0.0:
        return 0.0

    max_weight_dict = {1: 2, 2: 3, 3: 5}
    sx = harmonics.compute_cache(
        ker_base.n, max_weight_dict[order[0]], ker_base.is_singlet
    )
    sx_ns = sx.copy()
    if order[0] == 3 and (
        (backward_method != "" and ker_base.is_singlet)
        or (mode0 == 100 and mode1 == 100)
    ):
        # At N3LO for A_qq singlet or backward you need to compute
        # both the singlet and non-singlet like harmonics
        # avoiding recomputing all of them ...
        smx_ns = harmonics.smx(ker_base.n, np.array([s[0] for s in sx]), False)
        for w, sm in enumerate(smx_ns):
            sx_ns[w][-1] = sm
        sx_ns[2][2] = harmonics.S2m1(ker_base.n, sx[0][1], smx_ns[0], smx_ns[1], False)
        sx_ns[2][3] = harmonics.Sm21(ker_base.n, sx[0][0], smx_ns[0], False)
        sx_ns[3][5] = harmonics.Sm31(ker_base.n, sx[0][0], smx_ns[0], smx_ns[1], False)
        sx_ns[3][4] = harmonics.Sm211(ker_base.n, sx[0][0], sx[0][1], smx_ns[0], False)
        sx_ns[3][3] = harmonics.Sm22(
            ker_base.n, sx[0][0], sx[0][1], smx_ns[1], sx_ns[3][5], False
        )

    # compute the ome
    if ker_base.is_singlet or ker_base.is_QEDsinglet:
        indices = {21: 0, 100: 1, 90: 2}
        if is_polarized:
            if is_time_like:
                raise NotImplementedError("Polarized, time-like is not implemented")
            else:
                A = ome_ps.A_singlet(order, ker_base.n, sx, nf, L, is_msbar, sx_ns)
        else:
            if is_time_like:
                A = ome_ut.A_singlet(order, ker_base.n, sx, nf, L, is_msbar, sx_ns)
            else:
                A = ome_us.A_singlet(order, ker_base.n, sx, nf, L, is_msbar, sx_ns)
    else:
        indices = {200: 0, 91: 1}
        if is_polarized:
            if is_time_like:
                raise NotImplementedError("Polarized, time-like is not implemented")
            else:
                A = ome_ps.A_non_singlet(order, ker_base.n, sx, nf, L)
        else:
            if is_time_like:
                A = ome_ut.A_non_singlet(order, ker_base.n, sx, nf, L)
            else:
                A = ome_us.A_non_singlet(order, ker_base.n, sx, nf, L)

    # build the expansion in alpha_s depending on the strategy
    ker = build_ome(A, order, a_s, backward_method)

    # select the needed matrix element
    ker = ker[indices[mode0], indices[mode1]]

    # recombine everything
    return np.real(ker * integrand)


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
        matching scale
    is_backward: bool
        True for backward evolution
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
        super().__init__(config, managers, nf, q2, None)
        self.backward_method = config["backward_inversion"] if is_backward else ""
        if is_backward:
            self.is_intrinsic = True
        else:
            self.is_intrinsic = bool(len(config["intrinsic_range"]) != 0)
        self.L = L
        self.is_msbar = is_msbar
        # Note for the moment only QCD matching is implemented
        self.order = (self.order[0] - 1, self.order[1])

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
            if self.is_intrinsic or self.backward_method != "":
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
            if self.is_intrinsic or self.backward_method != "":
                labels.extend(
                    [
                        (21, br.matching_hplus_pid),
                        (100, br.matching_hplus_pid),
                        (br.matching_hplus_pid, br.matching_hplus_pid),
                    ]
                )
        return labels

    def quad_ker(self, label, logx, areas):
        """Return partially initialized integrand function.

        Parameters
        ----------
        label: tuple
            operator element pids
        logx: float
            Mellin inversion point
        areas : tuple
            basis function configuration

        Returns
        -------
        functools.partial
            partially initialized integration kernel
        """
        return functools.partial(
            quad_ker,
            order=self.order,
            mode0=label[0],
            mode1=label[1],
            is_log=self.int_disp.log,
            logx=logx,
            areas=areas,
            a_s=self.a_s,
            nf=self.nf,
            L=self.L,
            backward_method=self.backward_method,
            is_msbar=self.is_msbar,
            is_polarized=self.config["polarized"],
            is_time_like=self.config["time_like"],
        )

    @property
    def a_s(self):
        """Return the computed values for :math:`a_s`.

        Note that here you need to use :math:`a_s^{n_f+1}`
        """
        sc = self.managers["couplings"]
        return sc.a_s(
            self.sv_exponentiated_shift(self.q2_from), self.q2_from, nf_to=self.nf + 1
        )

    def compute(self):
        """Compute the actual operators (i.e. run the integrations)."""
        self.initialize_op_members()

        # At LO you don't need anything else
        if self.order[0] == 0:
            logger.info("%s: no need to compute matching at LO", self.log_label)
            return

        self.integrate()
