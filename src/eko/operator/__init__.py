# -*- coding: utf-8 -*-
r"""
This module contains the central operator classes.

See :doc:`Operator overview </code/Operators>`.
"""

import time
import logging

import numpy as np
from scipy import integrate
import numba as nb

from .. import mellin
from .. import interpolation
from .. import anomalous_dimensions as ad
from .. import beta
from ..kernels import non_singlet as ns
from ..kernels import singlet as s

from .member import OpMember
from .flavors import singlet_labels, full_labels

logger = logging.getLogger(__name__)


@nb.njit("c16[:](u1,string,c16,u1,f8)", cache=True)
def gamma_ns_fact(order, mode, n, nf, L):
    gamma_ns = ad.gamma_ns(order, mode[-1], n, nf)
    if order > 0:
        gamma_ns[1] -= beta.beta(0, nf) * gamma_ns[0] * L
    return gamma_ns


@nb.njit("c16[:,:,:](u1,c16,u1,f8)", cache=True)
def gamma_singlet_fact(order, n, nf, L):
    gamma_singlet = ad.gamma_singlet(
        order,
        n,
        nf,
    )
    if order > 0:
        gamma_singlet[1] -= beta.beta(0, nf) * gamma_singlet[0] * L
    return gamma_singlet


@nb.njit("f8(f8,u1,string,string,b1,f8,f8[:,:],f8,f8,f8,f8,u4,u1)", cache=True)
def quad_ker(
    u,
    order,
    mode,
    method,
    is_log,
    logx,
    areas,
    a1,
    a0,
    nf,
    L,
    ev_op_iterations,
    ev_op_max_order,
):
    """
    Raw kernel inside quad

    Parameters
    ----------
        u : float
            quad argument
        order : int
            perturbation order
        method : str
            method
        mode : str
            element in the singlet sector
        is_log : boolean
            logarithmic interpolation
        logx : float
            Mellin inversion point
        areas : tuple
            basis function configuration
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors
        L : float
            logarithmic ratio of factorization and renormalization scale
        ev_op_iterations : int
            number of evolution steps
        ev_op_max_order : int
            perturbative expansion order of U

    Returns
    -------
        ker : float
            evaluated integration kernel
    """
    is_singlet = mode[0] == "S"
    # get transformation to N integral
    if is_singlet:
        r, o = 0.4 * 16.0 / (1.0 - logx), 1.0
    else:
        r, o = 0.5, 0.0
    n = mellin.Talbot_path(u, r, o)
    jac = mellin.Talbot_jac(u, r, o)
    # check PDF is active
    if is_log:
        pj = interpolation.log_evaluate_Nx(n, logx, areas)
    else:
        pj = interpolation.evaluate_Nx(n, logx, areas)
    if pj == 0.0:
        return 0.0
    # compute the actual evolution kernel
    if is_singlet:
        gamma_singlet = gamma_singlet_fact(order, n, nf, L)
        ker = s.dispatcher(
            order, method, gamma_singlet, a1, a0, nf, ev_op_iterations, ev_op_max_order
        )
        # select element of matrix
        k = 0 if mode[2] == "q" else 1
        l = 0 if mode[3] == "q" else 1
        ker = ker[k, l]
    else:
        gamma_ns = gamma_ns_fact(order, mode, n, nf, L)
        ker = ns.dispatcher(
            order,
            method,
            gamma_ns,
            a1,
            a0,
            nf,
            ev_op_iterations,
        )
    # recombine everthing
    mellin_prefactor = np.complex(0.0, -1.0 / np.pi)
    return np.real(mellin_prefactor * ker * pj * jac)


class Operator:
    """
    Internal representation of a single EKO.

    The actual matrices are computed upon calling :meth:`compute`.
    :meth:`to_physical` will generate the :class:`PhysicalOperator` for the outside world.

    Parameters
    ----------
        master : eko.operator_grid.OperatorMaster
            the master instance
        q2_from : float
            evolution source
        q2_to : float
            evolution target
        mellin_cut : float
            cut to the upper limit in the mellin inversion
    """

    def __init__(self, config, managers, nf, q2_from, q2_to, mellin_cut=1e-2):
        self.config = config
        self.managers = managers
        self.nf = nf
        self.q2_from = q2_from
        self.q2_to = q2_to
        # TODO make 'cut' external parameter?
        self._mellin_cut = mellin_cut
        self.op_members = {}

    def labels(self):
        """
        Compute necessary sector labels to compute.

        Returns
        -------
            labels : list(str)
                sector labels
        """
        order = self.config["order"]
        labels = []
        # NS sector is dynamic
        if self.config["debug_skip_non_singlet"]:
            logger.warning("Evolution: skipping non-singlet sector")
        else:
            labels.append("NS_p")
            if order > 0:
                labels.append("NS_m")
        # singlet sector is fixed
        if self.config["debug_skip_singlet"]:
            logger.warning("Evolution: skipping singlet sector")
        else:
            labels.extend(singlet_labels)
        return labels

    def compute(self):
        """ compute the actual operators (i.e. run the integrations) """
        # Generic parameters
        int_disp = self.managers["interpol_dispatcher"]
        grid_size = len(int_disp.xgrid)

        # init all ops with zeros
        labels = self.labels()
        for n in full_labels:
            self.op_members[n] = OpMember(
                np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))
            )
        tot_start_time = time.perf_counter()
        # setup KernelDispatcher
        logger.info("Evolution: computing operators - 0/%d", grid_size)
        sc = self.managers["strong_coupling"]
        fact_to_ren = self.config["fact_to_ren"]
        a1 = sc.a_s(self.q2_to / fact_to_ren, self.q2_to)
        a0 = sc.a_s(self.q2_from / fact_to_ren, self.q2_from)
        # iterate output grid
        for k, logx in enumerate(np.log(int_disp.xgrid_raw)):
            start_time = time.perf_counter()
            # iterate basis functions
            for l, bf in enumerate(int_disp):
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
                            self.config["method"],
                            int_disp.log,
                            logx,
                            bf.areas_representation,
                            a1,
                            a0,
                            self.nf,
                            np.log(fact_to_ren),
                            self.config["ev_op_iterations"],
                            self.config["ev_op_max_order"],
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
                "Evolution: computing operators - %d/%d took: %f s",
                k + 1,
                grid_size,
                time.perf_counter() - start_time,
            )

        # closing comment
        logger.info("Evolution: Total time %f s", time.perf_counter() - tot_start_time)
        # copy non-singlet kernels, if necessary
        self.copy_ns_ops()

    def copy_ns_ops(self):
        """Copy non-singlet kernels, if necessary"""
        order = self.config["order"]
        if order == 0:  # in LO +=-=v
            for label in ["NS_v", "NS_m"]:
                self.op_members[label].value = self.op_members["NS_p"].value.copy()
                self.op_members[label].error = self.op_members["NS_p"].error.copy()
        elif order == 1:  # in NLO -=v
            self.op_members["NS_v"].value = self.op_members["NS_m"].value.copy()
            self.op_members["NS_v"].error = self.op_members["NS_m"].error.copy()
