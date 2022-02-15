# -*- coding: utf-8 -*-
r"""
This module contains the scale variation operator in scheme B
"""

import logging
import time

import numba as nb
import numpy as np
from scipy import integrate

from .. import anomalous_dimensions as ad
from .. import beta
from ..evolution_operator import Operator, QuadKerBase, select_singlet_element

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
    Raw scale variation B kernel inside quad.

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
    ker_base = QuadKerBase(u, is_log, logx, mode)
    integrand = ker_base.integrand(areas)
    if integrand == 0.0:
        return 0.0

    # compute the actual scale variation kernel
    if ker_base.is_singlet:
        gamma_singlet = ad.gamma_singlet(order, ker_base.n, nf)
        ker = singlet_dispatcher(gamma_singlet, a_s, order, nf, L)
        ker = select_singlet_element(ker, mode)
    else:
        gamma_ns = ad.gamma_ns(order, mode[-1], ker_base.n, nf)
        ker = non_singlet_dispatcher(gamma_ns, a_s, order, nf, L)

    # recombine everthing
    return np.real(ker * integrand)


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
        self.initialize_op_members()

        # At LO you don't need anything else
        if self.config["order"] == 0:
            logger.info("Scale Variation: no need to compute scale variations at LO")
            self.copy_ns_ops()
            return

        tot_start_time = time.perf_counter()
        a_s = self.strong_coupling.a_s(
            self.q2_from / self.fact_to_ren, fact_scale=self.q2_from
        )
        logger.info(
            "Scale Variation: (µ_F/µ_R)^2 = %e, Q^2 = %e, nf=%d",
            self.fact_to_ren,
            self.q2_from,
            self.nf,
        )
        logger.info("Scale Variation: computing operators - 0/%d", self.grid_size)
        # iterate output grid
        for k, logx in enumerate(np.log(self.int_disp.xgrid_raw)):
            start_time = time.perf_counter()
            # iterate basis functions
            for l, bf in enumerate(self.int_disp):
                if k == l and l == self.grid_size - 1:
                    continue
                # iterate sectors
                for label in self.labels:
                    # compute and set
                    res = integrate.quad(
                        quad_ker,
                        0.5,
                        1.0 - self._mellin_cut,
                        args=(
                            self.config["order"],
                            label,
                            self.int_disp.log,
                            logx,
                            bf.areas_representation,
                            a_s,
                            self.nf,
                            np.log(self.fact_to_ren),
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
                self.grid_size,
                time.perf_counter() - start_time,
            )

        # closing comment
        logger.info(
            "Scale Variation: Total time %f s", time.perf_counter() - tot_start_time
        )
        # copy non-singlet kernels, if necessary
        self.copy_ns_ops()
