# -*- coding: utf-8 -*-
"""
This module contains the main operator class.
"""

import time
import logging

import numpy as np
import numba as nb

from .. import mellin
from .. import interpolation
from .. import anomalous_dimensions as ad
from ..kernels import non_singlet as ns
from ..kernels import singlet as s

from .member import OpMember
from .physical import PhysicalOperator

logger = logging.getLogger(__name__)


@nb.njit
def compute_ns(order, mode, method, n, a1, a0, nf, ev_op_iterations):
    """
    Computes the non-singlet EKO

    Parameters
    ----------
        order : int
            perturbation order
        method : str
            method
        mode : str
            element in the non-singlet sector
        n : complex
            Mellin moment
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors
        ev_op_iterations : int
            number of evolution steps

    Returns
    -------
        e_ns : complex
            non-singlet EKO
    """
    # load data
    gamma_ns = ad.gamma_ns(order, mode[-1], n, nf)
    # switch by order and method
    return ns.dispatcher(
        order,
        method,
        gamma_ns,
        a1,
        a0,
        nf,
        ev_op_iterations,
    )


@nb.njit
def compute_singlet(
    order, mode, method, n, a1, a0, nf, ev_op_iterations, ev_op_max_order
):
    """
    Computes the singlet EKO

    Parameters
    ----------
        order : int
            perturbation order
        method : str
            method
        mode : str
            element in the singlet sector
        n : complex
            Mellin moment
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors
        ev_op_iterations : int
            number of evolution steps
        ev_op_max_order : int
            perturbative expansion order of U

    Returns
    -------
        e_s : numpy.ndarray
            singlet EKO
    """
    gamma_singlet = ad.gamma_singlet(
        order,
        n,
        nf,
    )
    ker = s.dispatcher(
        order, method, gamma_singlet, a1, a0, nf, ev_op_iterations, ev_op_max_order
    )
    # select element of matrix
    k = 0 if mode[2] == "q" else 1
    l = 0 if mode[3] == "q" else 1
    ker = ker[k, l]
    return ker


@nb.njit("f8(f8,u1,string,string,b1,f8,f8[:,:],f8,f8,u1,u4,u1)", cache=True)
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
        ker = compute_singlet(
            order, mode, method, n, a1, a0, nf, ev_op_iterations, ev_op_max_order
        )
    else:
        # ker = self.compute_ns(n)
        ker = compute_ns(order, mode, method, n, a1, a0, nf, ev_op_iterations)
    # recombine everthing
    mellin_prefactor = np.complex(0.0, -1.0 / np.pi)
    return np.real(mellin_prefactor * ker * pj * jac)


class Operator:
    """
    Internal representation of a single EKO.

    The actual matrices are computed only upon calling :meth:`compute`.
    :meth:`compose` will generate the :class:`PhysicalOperator` for the outside world.
    If not computed yet, :meth:`compose` will call :meth:`compute`.

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

    def compose(self, op_list, instruction_set, q2_final):
        """
        Compose all :class:`Operator` together.

        Calls :meth:`compute`, if necessary.

        Parameters
        ----------
            op_list : list(Operator)
                list of operators to merge
            instruction_set : dict
                list of instructions (generated by :class:`eko.thresholds.FlavourTarget`)
            q2_final : float
                final scale

        Returns
        -------
            op : PhysicalOperator
                final operator
        """
        # compute?
        if len(self.op_members.keys()) == 0:
            self.compute()
        # prepare operators
        op_to_compose = [self.op_members] + [i.op_members for i in reversed(op_list)]
        # iterate operators
        new_ops = {}
        for name, instructions in instruction_set:
            for origin, paths in instructions.items():
                key = f"{name}.{origin}"
                op = OpMember.join(op_to_compose, paths)
                # enforce new name
                op.name = key
                new_ops[key] = op
        return PhysicalOperator(new_ops, q2_final)

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
            labels.extend(["S_qq", "S_qg", "S_gq", "S_gg"])
        return labels

    def compute(self):
        """ compute the actual operators (i.e. run the integrations) """
        # Generic parameters
        int_disp = self.managers["interpol_dispatcher"]
        grid_size = len(int_disp.xgrid)

        # init all ops with zeros
        labels = self.labels()
        for n in ["S_qq", "S_qg", "S_gq", "S_gg", "NS_p", "NS_m", "NS_v"]:
            self.op_members[n] = OpMember(
                np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)), n
            )
        tot_start_time = time.perf_counter()
        # setup KernelDispatcher
        logger.info("Evolution: computing operators - 0/%d", grid_size)
        sc = self.managers["strong_coupling"]
        a1 = sc.a_s(self.q2_to)
        a0 = sc.a_s(self.q2_from)
        # iterate output grid
        for k, logx in enumerate(np.log(int_disp.xgrid_raw)):
            start_time = time.perf_counter()
            # iterate basis functions
            for l, bf in enumerate(int_disp):
                # iterate sectors
                for label in labels:
                    # compute and set
                    val, err = mellin.inverse_mellin_transform(
                        quad_ker,
                        self._mellin_cut,
                        [
                            self.config["order"],
                            label,
                            self.config["method"],
                            int_disp.log,
                            logx,
                            bf.areas_representation,
                            a1,
                            a0,
                            self.nf,
                            self.config["ev_op_iterations"],
                            self.config["ev_op_max_order"],
                        ],
                    )
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
