# -*- coding: utf-8 -*-
"""
This module defines the |OME| for the non-trivial matching conditions in the
|VFNS| evolution.
"""

import logging
import time

import numba as nb
import numpy as np
from scipy import integrate

from .. import basis_rotation as br
from .. import interpolation, mellin
from ..anomalous_dimensions import harmonics
from ..member import OpMember
from . import nlo, nnlo

logger = logging.getLogger(__name__)


@nb.njit("c16[:,:,:](u1,c16,c16[:],f8,b1)", cache=True)
def A_singlet(order, n, sx, L, is_msbar):
    r"""
    Computes the tower of the singlet |OME|.

    Parameters
    ----------
        order : int
            perturbative order
        n : complex
            Mellin variable
        sx : numpy.ndarray
            List of harmonic sums
        L : float
            :math:`log(q^2/m_h^2)`
        is_msbar: bool
            add the |MSbar| contribution

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
    if order[0] == 0:
        return np.zeros((1, 3, 3), np.complex_)
    A_singlet = np.zeros((order[0], 3, 3), np.complex_)
    if order[0] >= 1:
        A_singlet[0] = nlo.A_singlet_1(n, sx, L)
    if order[0] >= 2:
        A_singlet[1] = nnlo.A_singlet_2(n, sx, L, is_msbar)
    return A_singlet


@nb.njit("c16[:,:,:](u1,c16,c16[:],f8)", cache=True)
def A_non_singlet(order, n, sx, L):
    r"""
    Computes the tower of the non-singlet |OME|

    Parameters
    ----------
        order : int
            perturbative order
        n : complex
            Mellin variable
        sx : numpy.ndarray
            List of harmonic sums
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
    if order[0] == 0:
        return np.zeros((1, 2, 2), np.complex_)
    A_ns = np.zeros((order[0], 2, 2), np.complex_)
    if order[0] >= 1:
        A_ns[0] = nlo.A_ns_1(n, sx, L)
    if order[0] >= 2:
        A_ns[1] = nnlo.A_ns_2(n, sx, L)
    return A_ns


@nb.njit("c16[:,:](c16[:,:,:],u4,f8,string)", cache=True)
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
            empty or method for inverting the matching contidtion (exact or expanded)

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
    if backward_method == "expanded":
        # expended inverse
        if order[0] >= 1:
            ome -= a_s * A[0]
        if order[0] >= 2:
            ome += a_s**2 * (
                -A[1] + np.ascontiguousarray(A[0]) @ np.ascontiguousarray(A[0])
            )
    else:
        # forward or exact inverse
        if order[0] >= 1:
            ome += a_s * A[0]
        if order[0] >= 2:
            ome += a_s**2 * A[1]
        # need inverse exact ?  so add the missing pieces
        if backward_method == "exact":
            ome = np.linalg.inv(ome)
    return ome


@nb.njit("f8(f8,u1,u2,u2,b1,f8,f8[:,:],f8,f8,string,b1)", cache=True)
def quad_ker(
    u, order, mode0, mode1, is_log, logx, areas, a_s, L, backward_method, is_msbar
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
        L : float
            :math:`log(q^2/m_h^2)`
        backward_method : ["exact", "expanded" or ""]
            empty or method for inverting the matching contidtion (exact or expanded)
        is_msbar: bool
            add the |MSbar| contribution
    Returns
    -------
        ker : float
            evaluated integration kernel
    """
    is_singlet = mode0 in [100, 21, 90]
    # get transformation to N integral
    r = 0.4 * 16.0 / (1.0 - logx)
    if is_singlet:
        o = 1.0
        indices = {21: 0, 100: 1, 90: 2}
    else:
        o = 0.0
        indices = {200: 0, 91: 1}
    n = mellin.Talbot_path(u, r, o)
    jac = mellin.Talbot_jac(u, r, o)

    # compute the harmonics
    sx = np.zeros(3, np.complex_)
    if order[0] >= 1:
        sx = np.array([harmonics.harmonic_S1(n), harmonics.harmonic_S2(n)])
    if order[0] >= 2:
        sx = np.append(sx, harmonics.harmonic_S3(n))

    # compute the ome
    if is_singlet:
        A = A_singlet(order, n, sx, L, is_msbar)
    else:
        A = A_non_singlet(order, n, sx, L)

    # check PDF is active
    if is_log:
        pj = interpolation.log_evaluate_Nx(n, logx, areas)
    else:
        pj = interpolation.evaluate_Nx(n, logx, areas)

    if pj == 0.0:
        return 0.0

    # build the expansion in alpha_s depending on the strategy
    ker = build_ome(A, order, a_s, backward_method)

    # select the need matrix element
    ker = ker[indices[mode0], indices[mode1]]
    if ker == 0.0:
        return 0.0

    # recombine everthing
    mellin_prefactor = complex(0.0, -1.0 / np.pi)
    return np.real(mellin_prefactor * ker * pj * jac)


class OperatorMatrixElement:
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
        mellin_cut : float
            cut to the upper limit in the mellin inversion
    """

    def __init__(self, config, managers, is_backward, mellin_cut=1e-2):

        self.backward_method = config["backward_inversion"] if is_backward else ""
        self.is_intrinsic = bool(len(config["intrinsic_range"]) != 0)
        self.config = config
        self.sc = managers["strong_coupling"]
        self.int_disp = managers["interpol_dispatcher"]
        self._mellin_cut = mellin_cut
        self.ome_members = {}

    def labels(self):
        """
        Compute necessary sector labels to compute.

        Returns
        -------
            labels : list(str)
                sector labels
        """

        labels = []
        # non singlet labels
        if self.config["debug_skip_non_singlet"]:
            logger.warning("Matching: skipping non-singlet sector")
        else:
            labels.extend([(200, 200), (br.matching_hminus_pid, 200)])
            if self.is_intrinsic or self.backward_method != "":
                # intrisic labels, which are not zero at NLO
                labels.append((br.matching_hminus_pid, br.matching_hminus_pid))
                # if self.backward_method == "exact":
                #     # this contribution starts at NNLO, we don't have it for the moment
                #     labels.append("NS_qH")

        # same for singlet
        if self.config["debug_skip_singlet"]:
            logger.warning("Matching: skipping singlet sector")
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
                        (br.matching_hplus_pid, br.matching_hplus_pid),
                    ]
                )
                # if self.backward_method == "exact":
                #     labels.extend(["S_qH"])
        return labels

    def compute(self, q2, nf, L, is_msbar):
        """
        compute the actual operators (i.e. run the integrations)

        Parameters
        ----------
            q2: float
                threshold scale
            nf: int
                number of active flavor below threshold
            L: float
                log of K threshold squared
            is_msbar: bool
                add the |MSbar| contribution
        """
        # import pdb;pdb.set_trace()
        # init all ops with zeros
        grid_size = len(self.int_disp.xgrid)
        labels = self.labels()
        for n in labels:
            if n[0] == n[1]:
                self.ome_members[n] = OpMember(
                    np.eye(grid_size), np.zeros((grid_size, grid_size))
                )
            else:
                self.ome_members[n] = OpMember(
                    np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))
                )
        # import pdb; pdb.set_trace()
        # At LO you don't need anything else
        if self.config["orders"] == 0:
            logger.info("Matching: no need to compute matching at LO")
            self.copy_ome()
            return

        # Note that here you need to use a_s^{nf+1}(q2)
        a_s = self.sc.a(q2 / self.config["fact_to_ren"], q2, nf + 1)[0]

        tot_start_time = time.perf_counter()
        logger.info("Matching: computing operators - 0/%d", grid_size)
        # iterate output grid
        for k, logx in enumerate(np.log(self.int_disp.xgrid_raw)):
            start_time = time.perf_counter()
            # iterate basis functions
            for l, bf in enumerate(self.int_disp):
                if k == l and l == grid_size - 1:
                    continue
                # iterate sectors
                for label in labels:
                    # compute and set
                    res = integrate.quad(
                        quad_ker,
                        0.5,
                        1.0 - self._mellin_cut,
                        args=(
                            self.config["orders"],
                            label[0],
                            label[1],
                            self.int_disp.log,
                            logx,
                            bf.areas_representation,
                            a_s,
                            L,
                            self.backward_method,
                            is_msbar,
                        ),
                        epsabs=1e-12,
                        epsrel=1e-5,
                        limit=100,
                        full_output=1,
                    )
                    val, err = res[:2]
                    self.ome_members[label].value[k][l] = val
                    self.ome_members[label].error[k][l] = err

            logger.info(
                "Matching: computing operators - %d/%d took: %f s",
                k + 1,
                grid_size,
                time.perf_counter() - start_time,
            )

        # closing comment
        logger.info("Matching: Total time %f s", time.perf_counter() - tot_start_time)
        self.copy_ome()

    def copy_ome(self):
        """Add the missing |OME|, if necessary"""
        grid_size = len(self.int_disp.xgrid)
        # basic labels skipped with skip debug
        for label in [
            (200, 200),
            (br.matching_hplus_pid, 21),
            (br.matching_hplus_pid, 100),
            (br.matching_hminus_pid, 200),
            *br.singlet_labels,
        ]:
            if label not in self.ome_members:
                self.ome_members[label] = OpMember(
                    np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))
                )

        # intrinsic labels not computed yet
        if self.is_intrinsic:
            for label in [
                (100, br.matching_hplus_pid),
                (200, br.matching_hminus_pid),
                (br.matching_hminus_pid, br.matching_hminus_pid),
                (br.matching_hplus_pid, br.matching_hplus_pid),
                (21, br.matching_hplus_pid),
            ]:
                if label not in self.ome_members:
                    self.ome_members[label] = OpMember(
                        np.zeros((grid_size, grid_size)),
                        np.zeros((grid_size, grid_size)),
                    )
