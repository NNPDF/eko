# -*- coding: utf-8 -*-
"""
This module defines the KernelDispatcher and the actual integration kernel.
"""

import logging

import numpy as np

import eko.strong_coupling as sc
import eko.anomalous_dimensions as ad
from eko import mellin

from . import non_singlet as ns
from . import singlet as s

logger = logging.getLogger(__name__)


class IntegrationKernelObject:
    """
    Actual integration kernel.

    The object gets adjusted for each inversion point, basis function or sector.

    Parameters
    ----------
        kernel_dispatcher : eko.kernel_generation.KernelDispatcher
            parent dispatcher
    """

    def __init__(self, kernel_dispatcher):
        self.kernel_dispatcher = kernel_dispatcher
        self.pdf = self.kernel_dispatcher.interpol_dispatcher[0].callable
        self.mode = ""

    @property
    def is_singlet(self):
        """Are we currently in the singlet sector?"""
        return self.mode[0] == "S"

    def get_path_params(self):
        """
        Determine the Talbot parameters.

        Returns
        -------
            r,o : tuple(float)
                Talbot parameters
        """
        if self.is_singlet:
            return 0.4 * 16.0 / (1.0 - self.var("logx")), 1.0
        return 0.5, 0.0

    def var(self, name):
        """shortcut to the parent variables"""
        return self.kernel_dispatcher.var[name]

    def config(self, name):
        """shortcut to the parent config"""
        return self.kernel_dispatcher.config[name]

    def extra_args_ns(self):
        """additional arguments for the singlet EKO"""
        order = self.config("order")
        method = self.config("method")
        args = []
        if order == 1 and method in ["truncated", "ordered-truncated"]:
            args.append(self.config("ev_op_iterations"))
        return args

    def extra_args_s(self):
        """additional arguments for the non-singlet EKO"""
        order = self.config("order")
        method = self.config("method")
        args = []
        if order == 1:
            if method in [
                "truncated",
                "ordered-truncated",
                "iterate-exact",
                "iterate-expanded",
                "perturbative-exact",
                "perturbative-expanded",
            ]:
                args.append(self.config("ev_op_iterations"))
            if method in [
                "perturbative-exact",
                "perturbative-expanded",
            ]:
                args.append(self.config("ev_op_max_order"))
        return args

    def compute_ns(self, n):
        """
        Computes the non-singlet EKO

        Parameters
        ----------
            n : complex
                Mellin moment

        Returns
        -------
            e_ns : complex
                non-singlet EKO
        """
        order = self.config("order")
        # load data
        gamma_ns = ad.gamma_ns(
            order,
            self.mode[-1],
            n,
            self.var("nf"),
        )
        # switch order and method
        method = self.config("method")
        if order == 0:
            fnc = ns.dispatcher_lo(method)
        elif order == 1:
            fnc = ns.dispatcher_nlo(method)
        return fnc(
            gamma_ns,
            self.var("a1"),
            self.var("a0"),
            self.var("nf"),
            *self.extra_args_ns(),
        )

    def compute_singlet(self, n):
        """
        Computes the singlet EKO

        Parameters
        ----------
            n : complex
                Mellin moment

        Returns
        -------
            e_s : numpy.ndarray
                singlet EKO
        """
        order = self.config("order")
        gamma_singlet = ad.gamma_singlet(
            order,
            n,
            self.var("nf"),
        )
        method = self.config("method")
        if order == 0:
            fnc = s.dispatcher_lo(method)
        elif order == 1:
            fnc = s.dispatcher_nlo(method)
        return fnc(
            gamma_singlet,
            self.var("a1"),
            self.var("a0"),
            self.var("nf"),
            *self.extra_args_s(),
        )

    def __call__(self, u):
        """
        Called function under the integral.

        Parameters
        ----------
            u : float
                integration variable

        Returns
        -------
            ker : float
                kernel evaluated at `u`
        """
        # get transformation to N integral
        path_params = self.get_path_params()
        n = mellin.Talbot_path(u, *path_params)
        jac = mellin.Talbot_jac(u, *path_params)
        # check PDF is active
        pj = self.pdf(n, self.var("logx"), self.var("areas"))
        # print(self.pdf.inspect_types())
        if pj == 0.0:
            return 0.0
        # compute the actual evolution kernel
        if self.is_singlet:
            ker = self.compute_singlet(n)
            # select element of matrix
            k = 0 if self.mode[2] == "q" else 1
            l = 0 if self.mode[3] == "q" else 1
            ker = ker[k, l]
        else:
            ker = self.compute_ns(n)
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
        self.obj = IntegrationKernelObject(self)

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

    def init_loops(self, nf, a1, a0):
        """
        Called before the heavy grid-basis-functions-sectors loops.

        Parameters
        ----------
            nf : int
                number of flavors
            a1 : float
                strong coupling at target scale
            a0 : float
                strong coupling at initial scale
        """
        self.var["nf"] = nf
        self.var["a1"] = a1
        self.var["a0"] = a0
