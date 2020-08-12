# -*- coding: utf-8 -*-
"""
    This module contains functions that generate other functions
    All functions will receive a set of parameters so that the output
    function can be always numba compilable

    The generator functions `get_kernel_{kernel_type}` have as signature
    the following parameters:

        - `basis_function`: a callable function with a (N, lnx) signature
        - `nf` : number of flavours
        - `constants`: an instance of the Constants class
"""

from collections import abc
import logging

import numpy as np
import numba as nb

import eko.strong_coupling as sc
import eko.anomalous_dimensions.lo as ad_lo
import eko.mellin as mellin

logger = logging.getLogger(__name__)

class KernelDispatcher:
    """
        The kernel dispatcher does the common preparation for the kernel functions.

        Parameters
        ----------
            interpol_dispatcher : InterpolatorDispatcher
                An instance of the InterpolatorDispatcher class
            constants : Constants
                An instance of the Constants class
            order : int
                order in perturbation theory - ``0`` is leading order
            numba_it : bool  (default: True)
                If true, the functions will be `numba` compiled
    """

    def __init__(self, interpol_dispatcher, constants, order, method, numba_it=True):
        self.interpol_dispatcher = interpol_dispatcher
        self.constants = constants
        self.order = order
        self.method = method
        self.numba_it = numba_it
        self.kernels = {}

    @classmethod
    def from_dict(cls, setup, interpol_dispatcher, constants, numba_it=True):
        """
            Create the object from the theory dictionary.

            Read keys:

                - PTO : required, perturbative order
                - ModEv : optional, method to solve RGE, default=EXA

            Parameters
            ----------
                setup : dict
                    theory dictionary
                interpol_dispatcher : InterpolatorDispatcher
                    An instance of the InterpolatorDispatcher class
                constants : Constants
                    An instance of the Constants class
                numba_it : bool  (default: True)
                    If true, the functions will be `numba` compiled
        """
        order = setup["PTO"]
        mod_ev = setup.get("ModEv", "EXA")
        if mod_ev == "EXA":
            method = "exact"
        elif mod_ev == "EXP":
            method = "LL"
        elif mod_ev == "TRN":
            method = "truncated"
        else:
            raise ValueError(f"Unknown evolution mode {mod_ev}")
        return cls(interpol_dispatcher, constants, order, method, numba_it)

    def collect_singlets(self, nf, basis_function):
        r"""
            Returns the singlet integration kernels
            :math:`\ES{a_s^1}{a_s^0}`.

            Parameters
            ----------
                nf : int
                    number of active flavors
                basis_function : callable
                    accompainging basis function

            Returns
            -------
                ker : dict
                    (physical) kernels, which will be further modified for the
                    actual Mellin implementation
        """
        kers = {}
        CA = self.constants.CA
        CF = self.constants.CF
        beta_0 = sc.beta_0(nf, self.constants.CA, self.constants.CF, self.constants.TF)

        def get_ker(k, l):
            """(k,l)-th element of singlet kernel matrix"""

            def ker(N, lnx, a1, a0):
                """a singlet integration kernel"""
                l_p, l_m, e_p, e_m = ad_lo.get_Eigensystem_gamma_singlet_0(N, nf, CA, CF)
                ln_p = np.log(a1/a0) * l_p / beta_0
                ln_m = np.log(a1/a0) * l_m / beta_0
                interpoln = basis_function(N, lnx)
                return (e_p[k][l] * np.exp(ln_p) + e_m[k][l] * np.exp(ln_m)) * interpoln

            return ker

        kers["S_qq"] = get_ker(0, 0)
        kers["S_qg"] = get_ker(0, 1)
        kers["S_gq"] = get_ker(1, 0)
        kers["S_gg"] = get_ker(1, 1)
        return kers

    def collect_non_singlets(self, nf, basis_function):
        r"""
            Returns the non-singlet integration kernels
            :math:`\tilde E_{ns}(a_s^1 \leftarrow a_s^0)`.

            Parameters
            ----------
                nf : int
                    number of active flavors
                basis_function : callable
                    accompainging basis function

            Returns
            -------
                ker : dict
                    (physical) kernels, which will be further modified for the
                    actual Mellin implementation
        """
        kers = {}
        CA = self.constants.CA
        CF = self.constants.CF
        beta_0 = sc.beta_0(nf, self.constants.CA, self.constants.CF, self.constants.TF)

        def ker(n, lnx, a1, a0):
            """true non-siglet integration kernel"""
            ln = np.log(a1/a0) * ad_lo.gamma_ns_0(n, nf, CA, CF) / beta_0
            interpoln = basis_function(n, lnx)
            return np.exp(ln) * interpoln
        kers["NS_p"] = ker
        return kers

    def set_up_all_integrands(self, nf):
        """
            Compiles singlet and non-singlet integration kernel for each basis function.

            Parameters
            ----------
                nf : int
                    number of active flavors
        """
        # nothing to do?
        if nf in self.kernels:
            return
        # Setup path
        path, jac = mellin.get_path_Talbot()
        # iterate all basis functions and collect all functions in the sectors
        kernels_nf = []
        for basis_function in self.interpol_dispatcher:
            bf_kers = self.collect_singlets(nf, basis_function.callable)
            bf_kers.update(self.collect_non_singlets(nf, basis_function.callable))
            # compile
            for label, ker in bf_kers.items():
                bf_kers[label] = mellin.compile_integrand(self.njit(ker), path, jac, self.numba_it)
            kernels_nf.append(bf_kers)
        self.kernels[nf] = kernels_nf

    def njit(self, function):
        """
            Do nb.njit to the function if the `numba_it` flag is set to True.

            Checks whether a list of functions is passed.

            Parameters
            ---------
                function : callable or list(callable)
                    input (list of) callable(s)

            Returns
            -------
                function : callable or list(callable)
                    compiled (list of) callable(s)
        """
        if isinstance(function, abc.Iterable):
            return [self.njit(f) for f in function]
        if self.numba_it:
            return nb.njit(function)
        else:
            return function
