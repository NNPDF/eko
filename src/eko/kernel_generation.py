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

import logging

import numpy as np
import numba as nb

import eko.strong_coupling as sc
import eko.anomalous_dimensions.lo as ad_lo
import eko.anomalous_dimensions.nlo as ad_nlo
import eko.mellin as mellin

logger = logging.getLogger(__name__)


class KernelDispatcher:
    """
        Does the common preparation for the kernel functions.

        Parameters
        ----------
            interpol_dispatcher : InterpolatorDispatcher
                An instance of the InterpolatorDispatcher class
            constants : Constants
                An instance of the Constants class
            order : int
                order in perturbation theory - ``0`` is leading order
            method : str
                solution strategy
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

            Returns
            -------
                obj : cls
                    created object
        """
        order = int(setup["PTO"])
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

    def collect_kers(self, nf, basis_function):
        r"""
            Returns the integration kernels
            :math:`\tilde {\mathbf E}_{ns}(a_s^1 \leftarrow a_s^0)`.

            Parameters
            ----------
                nf : int
                    number of active flavors
                basis_function : callable
                    accompaying basis function

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
        order = self.order

        # provide the integrals
        j00 = self.njit(lambda a1, a0: np.log(a1 / a0) / beta_0)
        if order > 0:  # NLO constants and integrals
            beta_1 = sc.beta_1(
                nf, self.constants.CA, self.constants.CF, self.constants.TF
            )
            b1 = beta_1 / beta_0
            j11 = self.njit(
                lambda a1, a0: 1 / beta_1 * np.log((1 + a1 * b1) / (1 + a0 * b1))
            )
            j01 = self.njit(lambda a1, a0: j00(a1, a0) - b1 * j11(a1, a0))

        # singlet kernels
        def get_ker_s(k, l):
            """getter for (k,l)-th element of singlet kernel matrix"""

            def ker_s(N, lnx, a1, a0):
                """a singlet integration kernel"""
                # LO
                gamma_S_0 = ad_lo.gamma_singlet_0(N, nf, CA, CF)
                ln = gamma_S_0 * j00(a1, a0)
                # NLO
                if order > 0:
                    gamma_S_1 = ad_nlo.gamma_singlet_1(N, nf, CA, CF)
                    ln = gamma_S_0 * j01(a1, a0) + gamma_S_1 * j11(a1, a0)
                l_p, l_m, e_p, e_m = ad_lo.eigensystem_gamma_singlet_0(ln)
                e = e_m * np.exp(l_m) + e_p * np.exp(l_p)
                pdf = basis_function(N, lnx)
                return e[k][l] * pdf

            return ker_s

        kers["S_qq"] = get_ker_s(0, 0)
        kers["S_qg"] = get_ker_s(0, 1)
        kers["S_gq"] = get_ker_s(1, 0)
        kers["S_gg"] = get_ker_s(1, 1)

        # non-singlet kernels
        def get_ker_ns(mode):
            """getter for mode-flavored non-singlet kernel"""

            def ker_ns(n, lnx, a1, a0):
                """true non-siglet integration kernel"""
                # LO
                gamma_ns_0 = ad_lo.gamma_ns_0(n, nf, CA, CF)
                ln = gamma_ns_0 * j00(a1, a0)
                # NLO
                if order > 0:
                    if mode == "p":
                        gamma_ns_1 = ad_nlo.gamma_nsp_1(n, nf, CA, CF)
                    elif mode == "m":
                        gamma_ns_1 = ad_nlo.gamma_nsm_1(n, nf, CA, CF)
                    ln = gamma_ns_0 * j01(a1, a0) + gamma_ns_1 * j11(a1, a0)
                pdf = basis_function(n, lnx)
                return np.exp(ln) * pdf

            return ker_ns

        # in LO: +=-=v
        kers["NS_p"] = get_ker_ns("p")
        if order > 0:  # in NLO: -=v
            kers["NS_m"] = get_ker_ns("m")
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
            bf_kers = self.collect_kers(nf, basis_function.callable)
            # compile
            for label, ker in bf_kers.items():
                bf_kers[label] = mellin.compile_integrand(
                    self.njit(ker), path, jac, self.numba_it
                )
            kernels_nf.append(bf_kers)
        self.kernels[nf] = kernels_nf

    def njit(self, function):
        """
            Do nb.njit to the function if the `numba_it` flag is set to True.

            Parameters
            ---------
                function : callable
                    input callable

            Returns
            -------
                function : callable
                    compiled callable
        """
        if self.numba_it:
            return nb.njit(function)
        else:
            return function
