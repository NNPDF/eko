# -*- coding: utf-8 -*-
"""
This module defines the |OME| for the non-trivial matching conditions in the
|VFNS| evolution.
"""

import functools
import logging

import numba as nb
import numpy as np

from .. import basis_rotation as br
from .. import harmonics
from ..evolution_operator import Operator, QuadKerBase
from . import as1, as2

# from . import as1, as2, as3

logger = logging.getLogger(__name__)


@nb.njit(cache=True)
def compute_harmonics_cache(n, order, is_singlet):
    r"""
    Get the harmonics sums cache

    Parameters
    ----------
        n: complex
            Mellin moment
        order: int
            perturbative order
        is_singlet: bool
            symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N = 1`),
            False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
        sx: list
            harmonic sums cache. At |N3LO| it contains:

            .. math ::
                [[S_1,S_{-1}],
                [S_2,S_{-2}],
                [S_{3}, S_{2,1}, S_{2,-1}, S_{-2,1}, S_{-2,-1}, S_{-3}],
                [S_{4}, S_{3,1}, S_{2,1,1}, S_{-2,-2}, S_{-3, 1}, S_{-4}],]

    """
    # max harmonics sum weight for each qcd order
    max_weight = {1: 2, 2: 3, 3: 5}
    # max number of harmonics sum of a given weight for each qcd order
    n_max_sums_weight = {1: 1, 2: 3, 3: 7}
    sx = harmonics.base_harmonics_cache(
        n, is_singlet, max_weight[order], n_max_sums_weight[order]
    )
    if order == 2:
        # Add Sm21 to cache
        sx[2, 1] = harmonics.Sm21(n, sx[0, 0], sx[0, -1], is_singlet)
    if order == 3:
        # Add weight 3 and 4 to cache
        sx[2, 1:-2] = harmonics.s3x(n, sx[:, 0], sx[:, -1], is_singlet)
        sx[3, 1:-1] = harmonics.s4x(n, sx[:, 0], sx[:, -1], is_singlet)
    # return list of list keeping the non zero values
    return [[el for el in sx_list if el != 0] for sx_list in sx]


@nb.njit(cache=True)
def A_singlet(order, n, sx, nf, L, is_msbar, sx_ns=None):
    r"""
    Computes the tower of the singlet |OME|.

    Parameters
    ----------
        order : int
            perturbative order
        n : complex
            Mellin variable
        sx : list
            singlet like harmonic sums cache
        nf: int
            number of active flavor below threshold
        L : float
            :math:`log(q^2/m_h^2)`
        is_msbar: bool
            add the |MSbar| contribution
        sx_ns : list
            non-singlet like harmonic sums cache

    Returns
    -------
        A_singlet : numpy.ndarray
            singlet |OME|

    See Also
    --------
        eko.matching_conditions.nlo.A_singlet_1 : :math:`A^{S,(1)}(N)`
        eko.matching_conditions.nlo.A_hh_1 : :math:`A_{HH}^{(1)}(N)`
        eko.matching_conditions.nlo.A_gh_1 : :math:`A_{gH}^{(1)}(N)`
        eko.matching_conditions.nnlo.A_singlet_2 : :math:`A_{S,(2)}(N)`
    """
    A_s = np.zeros((order, 3, 3), np.complex_)
    if order >= 1:
        A_s[0] = as1.A_singlet(n, sx, L)
    if order >= 2:
        A_s[1] = as2.A_singlet(n, sx, L, is_msbar)
    # if order >= 3:
    #     A_s[2] = as3.A_singlet(n, sx, sx_ns, nf, L)
    return A_s


@nb.njit(cache=True)
def A_non_singlet(order, n, sx, nf, L):
    r"""
    Computes the tower of the non-singlet |OME|

    Parameters
    ----------
        order : int
            perturbative order
        n : complex
            Mellin variable
        sx : list
            harmonic sums cache
        nf: int
            number of active flavor below threshold
        L : float
            :math:`log(q^2/m_h^2)`

    Returns
    -------
        A_non_singlet : numpy.ndarray
            non-singlet |OME|

    See Also
    --------
        eko.matching_conditions.nlo.A_hh_1 : :math:`A_{HH}^{(1)}(N)`
        eko.matching_conditions.nnlo.A_ns_2 : :math:`A_{qq,H}^{NS,(2)}`
    """
    A_ns = np.zeros((order, 2, 2), np.complex_)
    if order >= 1:
        A_ns[0] = as1.A_ns(n, sx, L)
    if order >= 2:
        A_ns[1] = as2.A_ns(n, sx, L)
    # if order >= 3:
    #     A_ns[2] = as3.A_ns(n, sx, nf, L)
    return A_ns


@nb.njit(cache=True)
def build_ome(A, order, a_s, backward_method):
    r"""
    Construct the matching expansion in :math:`a_s` with the appropriate method.

    Parameters
    ----------
        A : numpy.ndarray
            list of |OME|
        order : int
            perturbation order
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
        if order >= 1:
            ome -= a_s * A[0]
        if order >= 2:
            ome += a_s**2 * (-A[1] + A[0] @ A[0])
        if order >= 3:
            ome += a_s**3 * (-A[2] + A[0] @ A[1] + A[1] @ A[0] - A[0] @ A[0] @ A[0])
    else:
        # forward or exact inverse
        if order >= 1:
            ome += a_s * A[0]
        if order >= 2:
            ome += a_s**2 * A[1]
        if order >= 3:
            ome += a_s**3 * A[2]
        # need inverse exact ?  so add the missing pieces
        if backward_method == "exact":
            ome = np.linalg.inv(ome)
    return ome


@nb.njit(cache=True)
def quad_ker(
    u, order, mode0, mode1, is_log, logx, areas, a_s, nf, L, backward_method, is_msbar
):
    """
    Raw kernel inside quad

    Parameters
    ----------
        u : float
            quad argument
        order : int
            perturbation order
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
            :math:`log(q^2/m_h^2)`
        backward_method : ["exact", "expanded" or ""]
            empty or method for inverting the matching condition (exact or expanded)
        is_msbar: bool
            add the |MSbar| contribution
    Returns
    -------
        ker : float
            evaluated integration kernel
    """
    ker_base = QuadKerBase(u, is_log, logx, mode0)
    integrand = ker_base.integrand(areas)
    if integrand == 0.0:
        return 0.0

    sx = compute_harmonics_cache(ker_base.n, order, ker_base.is_singlet)
    sx_ns = None
    if order == 3 and (
        (backward_method != "" and ker_base.is_singlet)
        or (mode0 == 100 and mode0 == 100)
    ):
        # At N3LO for A_qq singlet or backward you need to compute
        # both the singlet and non-singlet like harmonics
        # avoiding recomputing all of them ...
        sx_ns = sx.copy()
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
    if ker_base.is_singlet:
        indices = {21: 0, 100: 1, 90: 2}
        A = A_singlet(order, ker_base.n, sx, nf, L, is_msbar, sx_ns)
    else:
        indices = {200: 0, 91: 1}
        A = A_non_singlet(order, ker_base.n, sx, nf, L)

    # build the expansion in alpha_s depending on the strategy
    ker = build_ome(A, order, a_s, backward_method)

    # select the needed matrix element
    ker = ker[indices[mode0], indices[mode1]]

    # recombine everthing
    return np.real(ker * integrand)


class OperatorMatrixElement(Operator):
    """
    Internal representation of a single |OME|.

    The actual matrices are computed upon calling :meth:`compute`.

    Parameters
    ----------
        config : dict
            configuration
        managers : dict
            managers
        is_backward: bool
            True for backward evolution
        q2: float
            matching scale
        nf: int
            number of active flavor below threshold
        L: float
            log of K threshold squared
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

    def __init__(self, config, managers, nf, q2, is_backward, L, is_msbar):
        super().__init__(config, managers, nf, q2)
        self.backward_method = config["backward_inversion"] if is_backward else ""
        if is_backward:
            self.is_intrinsic = True
        else:
            self.is_intrinsic = bool(len(config["intrinsic_range"]) != 0)
        self.L = L
        self.is_msbar = is_msbar

    @property
    def labels(self):
        """
        Compute necessary sector labels to compute.

        Returns
        -------
            labels : list(str)
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
        """
        Partially initialized integrand function

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
            quad_ker : functools.partial
                partially initialized intration kernel

        """
        return functools.partial(
            quad_ker,
            order=self.config["order"],
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
        )

    @property
    def a_s(self):
        """
        Returns the computed values for :math:`a_s`.
        Note that here you need to use :math:`a_s^{n_f+1}`
        """
        sc = self.managers["strong_coupling"]
        return sc.a_s(self.q2_from / self.fact_to_ren, self.q2_from, nf_to=self.nf + 1)

    def compute(self):
        """
        compute the actual operators (i.e. run the integrations)
        """
        self.initialize_op_members()

        # At LO you don't need anything else
        if self.config["order"] == 0:
            logger.info("%s: no need to compute matching at LO", self.log_label)
            return

        self.integrate()
