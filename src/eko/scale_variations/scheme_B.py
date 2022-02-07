# -*- coding: utf-8 -*-
r"""
This module contains the scale varion operator in scheme B
"""

import logging
import time

import numba as nb
import numpy as np
from scipy import integrate

from ..evolution_operator import Operator
from .. import anomalous_dimensions as ad
from .. import beta, interpolation, mellin
from ..basis_rotation import full_labels
from ..member import OpMember

logger = logging.getLogger(__name__)


@nb.njit("c16(c16[:],f8,u1,u1,f8)", cache=True)
def non_singlet_dispatcher(gamma, a_s, order, nf, L):
    """
    Scale Variation non singlet dispatcher

    Parameters
    ----------
        gamma : numpy.ndarray
            anomalous dimensions
        a_s :  float
            target coupling value
        order : int
            perturbation order
        nf : int
            number of active flavors
        L : float
            logarithmic ratio of factorization and renormalization scale

    Returns
    -------
        sv_ker : numpy.ndarray
            scale varion kernel
    """
    sv_ker = 1.0
    if order >= 1:
        sv_ker -= a_s * L * gamma[0]
    if order >= 2:
        sv_ker += a_s ** 2 * (
            -gamma[1] * L
            + 1 / 2 * (beta.beta_0(nf) * gamma[0] + gamma[0] * gamma[0]) * L ** 2
        )
    return sv_ker


@nb.njit("c16[:,:](c16[:,:,:],f8,u1,u1,f8)", cache=True)
def singlet_dispatcher(gamma, a_s, order, nf, L):
    """
    Scale Variation singlet dispatcher

    Parameters
    ----------
        gamma : numpy.ndarray
            anomalous dimensions
        a_s :  float
            target coupling value
        order : int
            perturbation order
        nf : int
            number of active flavors
        L : float
            logarithmic ratio of factorization and renormalization scale

    Returns
    -------
        sv_ker : numpy.ndarray
            scale varion kernel
    """
    sv_ker = np.eye(2, dtype=np.complex_)
    if order >= 1:
        sv_ker -= a_s * L * gamma[0]
    if order >= 2:
        sv_ker += a_s ** 2 * (
            -gamma[1] * L
            + 1
            / 2
            * (
                beta.beta_0(nf) * gamma[0]
                + np.ascontiguousarray(gamma[0]) @ np.ascontiguousarray(gamma[0])
            )
            * L ** 2
        )
    return sv_ker


@nb.njit("f8(f8,u1,string,b1,f8,f8[:,:],f8,u1,f8)", cache=True)
def quad_ker(
    u,
    order,
    mode,
    is_log,
    logx,
    areas,
    a_s,
    nf,
    L,
):
    """
    Raw kernel inside quad.

    Parameters
    ----------
        u : float
            quad argument
        order : int
            perturbation order
        mode : str
            sector element
        is_log : boolean
            is a logarithmic interpolation
        logx : float
            Mellin inversion point
        areas : tuple
            basis function configuration
        a_s : float
            target coupling value
        nf : int
            number of active flavors
        L : float
            logarithm of the squared ratio of factorization and renormalization scale

    Returns
    -------
        ker : float
            evaluated scale variation kernel
    """
    # TODO: this code is reperated twice or 3 times
    is_singlet = mode[0] == "S"
    # get transformation to N integral
    if logx == 0.0:
        return 0.0
    r = 0.4 * 16.0 / (-logx)
    if is_singlet:
        o = 1.0
    else:
        o = 0.0
    n = mellin.Talbot_path(u, r, o)
    jac = mellin.Talbot_jac(u, r, o)
    # check PDF is active
    if is_log:
        pj = interpolation.log_evaluate_Nx(n, logx, areas)
    else:
        pj = interpolation.evaluate_Nx(n, logx, areas)
    if pj == 0.0:
        return 0.0

    # compute the actual scale variation kernel
    if is_singlet:
        gamma_singlet = ad.gamma_singlet(order, n, nf)
        ker = singlet_dispatcher(gamma_singlet, a_s, order, nf, L)
        # select element of matrix
        k = 0 if mode[2] == "q" else 1
        l = 0 if mode[3] == "q" else 1
        ker = ker[k, l]
    else:
        gamma_ns = ad.gamma_ns(order, mode[-1], n, nf)
        ker = non_singlet_dispatcher(gamma_ns, a_s, order, nf, L)

    # recombine everthing
    mellin_prefactor = complex(0.0, -1.0 / np.pi)
    return np.real(mellin_prefactor * ker * pj * jac)


class ScaleVariationOperator(Operator):
    """
    Internal representation of a Scale Variation operator.

    The actual matrices are computed upon calling :meth:`compute`.

    Parameters
    ----------
        config : dict
            configuration
        managers : dict
            managers
        nf : int
            number of active flavors
        q2 : float
            evolution scale
    """

    def __init__(self, config, managers, nf, q2):
        super().__init__(config, managers, nf, q2)

    def compute(self):
        """compute the actual operators (i.e. run the integrations)"""
        # TODO: here there are a lot of repetitions, you need to generalize
        # the evolution operator
        # Generic parameters
        int_disp = self.managers["interpol_dispatcher"]
        grid_size = len(int_disp.xgrid)

        # init all ops with identity or zeros if we skip them
        labels = self.labels()
        eye = OpMember(np.eye(grid_size), np.zeros((grid_size, grid_size)))
        zero = OpMember(*[np.zeros((grid_size, grid_size))] * 2)
        for n in full_labels:
            if n in labels:
                # off diag singlet are zero
                if n in ["S_qg", "S_gq"]:
                    self.op_members[n] = zero.copy()
                else:
                    self.op_members[n] = eye.copy()
            else:
                self.op_members[n] = zero.copy()

        # At LO you don't need anything else
        if self.config["order"] == 0:
            logger.info("Scale Variation: no need to compute scale variations at LO")
            self.copy_ns_ops()
            return

        tot_start_time = time.perf_counter()
        # setup ingredients
        sc = self.managers["strong_coupling"]
        fact_to_ren = self.config["fact_to_ren"]
        a_s = sc.a_s(self.q2_from / fact_to_ren, fact_scale=self.q2_from, nf_to=self.nf)
        logger.info(
            "Scale Variation: (µ_F/µ_R)^2 = %e, Q^2 = %e, nf=%d",
            fact_to_ren,
            self.q2_from,
            self.nf,
        )
        logger.info("Scale Variation: computing operators - 0/%d", grid_size)
        # iterate output grid
        for k, logx in enumerate(np.log(int_disp.xgrid_raw)):
            start_time = time.perf_counter()
            # iterate basis functions
            for l, bf in enumerate(int_disp):
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
                            self.config["order"],
                            label,
                            int_disp.log,
                            logx,
                            bf.areas_representation,
                            a_s,
                            self.nf,
                            np.log(fact_to_ren),
                        ),
                        epsabs=1e-12,
                        epsrel=1e-5,
                        limit=100,
                        full_output=1,
                    )
                    val, err = res[:2]
                    self.op_members[label].value[k][l] = val
                    self.op_members[label].error[k][l] = err

            logger.info(
                "Scale Variation: computing operators - %d/%d took: %f s",
                k + 1,
                grid_size,
                time.perf_counter() - start_time,
            )

        # closing comment
        logger.info(
            "Scale Variation: Total time %f s", time.perf_counter() - tot_start_time
        )
        # copy non-singlet kernels, if necessary
        self.copy_ns_ops()
