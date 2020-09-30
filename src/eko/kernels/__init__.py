# -*- coding: utf-8 -*-
"""
This module defines the KernelDispatcher and the actual integration kernel.
"""

import logging

import numpy as np
import numba as nb

from .. import strong_coupling as sc
from .. import anomalous_dimensions as ad
from .. import mellin
from .. import interpolation

from . import non_singlet as ns
from . import singlet as s

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


@nb.njit("f8(f8,u1,string,string,b1,f8,f8[:,:],f8,f8,u1,u4,u1)")
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


class KernelDispatcher:
    """
    Does the common preparation for the kernel functions.

    Parameters
    ----------
        config : dict
            configuration
        interpol_dispatcher : eko.interpolation.InterpolatorDispatcher
            the basis functions
    """

    def __init__(self, config, interpol_dispatcher):
        # check
        order = int(config["order"])
        method = config["method"]
        if not method in [
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
        if order == 0 and method != "iterate-exact":
            logger.warning("Kernels: In LO we use the exact solution always!")
        self.config = config
        # set managers
        self.interpol_dispatcher = interpol_dispatcher
        # init objects
        self.var = {}

    @classmethod
    def from_dict(cls, setup, interpol_dispatcher):
        """
        Create the object from the theory dictionary.

        Read keys:

            - PTO : required, perturbative order
            - ModEv : optional, method to solve RGE, default=EXA=iterate-exact

        Parameters
        ----------
            setup : dict
                theory dictionary
            interpol_dispatcher : InterpolatorDispatcher
                An instance of the InterpolatorDispatcher class

        Returns
        -------
            obj : cls
                created object
        """
        config = {}
        config["order"] = int(setup["PTO"])
        method = setup.get("ModEv", "iterate-exact")
        mod_ev2method = {
            "EXA": "iterate-exact",
            "EXP": "iterate-expanded",
            "TRN": "truncated",
        }
        method = mod_ev2method.get(method, method)
        config["method"] = method
        config["ev_op_max_order"] = setup.get("ev_op_max_order", 10)
        config["ev_op_iterations"] = setup.get("ev_op_iterations", 10)
        return cls(config, interpol_dispatcher)
