# -*- coding: utf-8 -*-
"""
This module defines the operator matrix elements for the non-trivial matching conditions in the
|VFNS| evolution.
"""

import time
import logging

import numpy as np
from scipy import integrate
import numba as nb

from .. import mellin
from .. import interpolation
from ..member import OpMember

from .nlo import A_gh_1, A_hh_1
from .nnlo import A_singlet_2, A_ns_2
from ..anomalous_dimensions import harmonics
from ..member import singlet_labels


logger = logging.getLogger(__name__)


@nb.njit("c16[:,:](c16,c16[:],u4,f8,f8,string,b1)", cache=True)
def build_singlet_ome(n, sx, order, a_s, L, backward_method, is_intrisinc):
    """Singlet matching matrix"""
    ker = np.eye(3, dtype=np.complex_)
    if backward_method == "expanded":
        if order >= 1:
            ker1 = np.zeros(2)  # TODO: A_singlet_1(n, sx, L) not ready yet
            ker[:-1, :-1] -= a_s * ker1
            if is_intrisinc:
                # intrisic h+ contribution
                ker[2, 2] -= a_s * A_hh_1(n, sx, L)
                ker[1, 2] -= a_s * A_gh_1(n, L)
        if order >= 2:
            ker[:-1, :-1] += a_s ** 2 * (-A_singlet_2(n, sx) + ker1 @ ker1)
    else:
        # forward or exact inverse
        if order >= 1:
            ker[:-1, :-1] += a_s * np.zeros(2)  # A_singlet_1(n, sx, L) not ready yet
            if is_intrisinc:
                # intrisic h+ contribution
                ker[2, 2] += a_s * A_hh_1(n, sx, L=0.0)
                ker[1, 2] += a_s * A_gh_1(n, L=0.0)
        if order >= 2:
            ker2 = a_s ** 2 * A_singlet_2(n, sx)
            ker[:-1, :-1] += ker2
            ker[2, 0] += ker2[0, 0]
            ker[2, 1] += ker2[0, 1]
        # need inverse exact ?, so add the missing pieces
        if backward_method == "exact":
            ker = np.linalg.inv(ker)
    return ker


@nb.njit("c16[:,:](c16,c16[:],u4,f8,f8,string,b1)", cache=True)
def build_non_singlet_ome(n, sx, order, a_s, L, backward_method, is_intrisinc):
    """Non singlet matching matrix"""
    ker = np.eye(2, dtype=np.complex_)
    if backward_method == "expanded":
        if order >= 1:
            ker1 = 0.0  # TODO: A_ns_1(n, sx, L) not ready yet
            ker[0, 0] -= a_s * ker1
            if is_intrisinc:
                # intrisic h+ contribution
                ker[1, 1] -= a_s * A_hh_1(n, sx, L)
        if order >= 2:
            ker[0, 0] += a_s ** 2 * (-A_ns_2(n, sx) + ker1 * ker1)
    else:
        # forward or exact inverse
        if order >= 1:
            ker[0, 0] += a_s * 0.0  # A_ns_1(n, sx, L) not ready yet
            if is_intrisinc:
                # intrisic h+ contribution
                ker[1, 1] += a_s * A_hh_1(n, sx, L)
        if order >= 2:
            ker2 = a_s ** 2 * A_ns_2(n, sx)
            ker[0, 0] += ker2
            ker[1, 0] += ker2
        # need inverse exact ?  so add the missing pieces
        if backward_method == "exact":
            ker = np.linalg.inv(ker)
    return ker


@nb.njit("f8(f8,u1,string,b1,f8,f8[:,:],f8,f8,string,b1)", cache=True)
def quad_ker(
    u, order, mode, is_log, logx, areas, a_s, L, backward_method, is_intrisinc
):
    """
    Raw kernel inside quad

    Parameters
    ----------
        u : float
            quad argument
        order : int
            perturbation order
        mode : str
            element in the singlet sector
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
        backward_method : [exact, expanded or None]
            None or method for inverting the matching contidtion (exact or expanded)
        is_intrinsic: bool
            True for intrinsic evolution
    Returns
    -------
        ker : float
            evaluated integration kernel
    """

    is_singlet = mode[0] == "S"
    # get transformation to N integral
    if is_singlet:
        r, o = 0.4 * 16.0 / (1.0 - logx), 1.0
        indeces = {"q": 0, "g": 1, "H": 2}
    else:
        r, o = 0.5, 0.0
        indeces = {"q": 0, "H": 1}

    n = mellin.Talbot_path(u, r, o)
    jac = mellin.Talbot_jac(u, r, o)
    # check PDF is active
    if is_log:
        pj = interpolation.log_evaluate_Nx(n, logx, areas)
    else:
        pj = interpolation.evaluate_Nx(n, logx, areas)

    if pj == 0.0:
        return 0.0

    # compute the harmonics
    sx = np.array(
        [harmonics.harmonic_S1(n), harmonics.harmonic_S2(n), harmonics.harmonic_S3(n)]
    )

    # compute the actual evolution kernel
    ker = 0.0 + 0j
    if is_singlet:
        ker = build_singlet_ome(n, sx, order, a_s, L, backward_method, is_intrisinc)
    else:
        ker = build_non_singlet_ome(n, sx, order, a_s, L, backward_method, is_intrisinc)

    ker = ker[indeces[mode[-2]], indeces[mode[-1]]]

    # recombine everthing
    mellin_prefactor = complex(0.0, -1.0 / np.pi)
    return np.real(mellin_prefactor * ker * pj * jac)


class OperatorMatrixElement:
    """
    Internal representation of a single Operator Matrix Element.

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
        self.order = config["order"]
        self.sc = managers["strong_coupling"]
        self.fact_to_ren = config["fact_to_ren"]
        self.int_disp = managers["interpol_dispatcher"]
        self._mellin_cut = mellin_cut
        self.ome_members = {}

    def compute(self, q2, mh2):
        """
        compute the actual operators (i.e. run the integrations)

        Parameters
        ----------
            q2: float
                threshold scale
            mh2: float
                heavy quark mass
        """

        # init all ops with zeros
        grid_size = len(self.int_disp.xgrid)
        # TODO: improve labels: NS_Hq and S_Hg are not really needed for
        #  forward evolution since are equal to NS_qq and S_qg.
        labels = ["S_Hg", "S_Hq", "NS_Hq", "NS_qq", *singlet_labels]
        if self.is_intrinsic:
            # intrisic labels
            labels.extend(
                [
                    "S_qH",  # starts at NNLO we don't have it for the moment
                    "S_gH",
                    "S_HH",
                    "NS_qH",  # starts at NNLO we don't have it for the moment
                    "NS_HH",
                ]
            )
        for n in labels:
            self.ome_members[n] = OpMember(
                np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))
            )
        L = np.log(q2 / mh2)
        a_s = self.sc.a_s(q2 / self.fact_to_ren, q2)

        # we always need to compute the opartors, since some are identies and others
        # are zeros.
        # if self.order <= 1 and self.is_intrinsic is False and L == 0.0:
        #     logger.info(
        #         "Matching: only trivial conditions are needed at PTO = %d", self.order
        #     )
        #     return

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
                            self.order,
                            label,
                            self.int_disp.log,
                            logx,
                            bf.areas_representation,
                            a_s,
                            L,
                            self.backward_method,
                            self.is_intrinsic,
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
