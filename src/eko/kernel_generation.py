"""
    This module contains functions that generate other functions
    All functions will receive a set of parameters so that the output
    function can be always numba compilable

    The generator functions `get_kernel_{kernel_type}` have as signature
    the following parameters:

        - `basis_function`: a callable function with a (N, lnx) signature
        - `nf` : number of flavours
        - `constants`: an instance of the Constants class
        - `beta_0` : value of `beta_0`
        - `delta_t`: value of `delta_t`

"""

from collections import abc
import numpy as np
import mpmath as mp
import numba as nb

import eko.alpha_s as alpha_s
import eko.splitting_functions_LO as sf_LO


def get_kernel_ns(basis_function, nf, constants, beta_0, delta_t):
    """Returns the non-singlet integration kernel"""
    CA = constants.CA
    CF = constants.CF

    def ker(n, lnx):
        """true non-siglet integration kernel"""
        ln = -delta_t * sf_LO.gamma_ns_0(n, nf, CA, CF) / beta_0
        interpoln = basis_function(n, lnx)
        return mp.exp(ln) * interpoln

    return ker


def get_kernels_s(basis_function, nf, constants, beta_0, delta_t):
    """Return all singlet integration kernels"""
    CA = constants.CA
    CF = constants.CF

    def get_ker(k, l):
        """true singlet kernel"""

        def ker(N, lnx):  # TODO here we are repeating too many things!
            """a singlet integration kernel"""
            l_p, l_m, e_p, e_m = sf_LO.get_Eigensystem_gamma_singlet_0(N, nf, CA, CF)
            ln_p = -delta_t * l_p / beta_0
            ln_m = -delta_t * l_m / beta_0
            interpoln = basis_function(N, lnx)
            return (e_p[k,l] * mp.exp(ln_p) + e_m[k,l] * mp.exp(ln_m)) * interpoln

        return ker

    return get_ker(0, 0), get_ker(0, 1), get_ker(1, 0), get_ker(1, 1)


class KernelDispatcher:
    """
        The kernel dispatcher does the common preparation for the kernel functions

        Upon calling the appropiate `compile_` method the dispatcher will return
        a `numba` compiled kernel.

        Parameters
        ----------
            interpol_dispatcher : InterpolatorDispatcher
                An instance of the InterpolatorDispatcher class
            constants : Constants
                An instance of the Constants class
            nf : float
                Number of flavour to consider
            delta_t : float
                Value of delta_t for this kernel
            numba_it : bool  (default: True)
                If true, the functions will be `numba` compiled
    """

    def __init__(self, interpol_dispatcher, constants, nf, delta_t, numba_it=False):
        self.interpol_dispatcher = interpol_dispatcher
        self.delta_t = delta_t
        self.nf = nf
        self.constants = constants
        self.beta_0 = alpha_s.beta_0(nf, constants.CA, constants.CF, constants.TF)
        self.numba_it = numba_it

    def _compiler(self, generating_function):
        """
            Call `generating_function` with the appropiate parameters
            and pass the output through numba
        """
        kernels = []
        for basis_function in self.interpol_dispatcher:
            new_ker = generating_function(
                basis_function.callable,
                self.nf,
                self.constants,
                self.beta_0,
                self.delta_t,
            )
            kernels.append(self.njit(new_ker))
        return kernels

    def compile_singlet(self):
        """Compiles the singlet integration kernels for each basis """
        return self._compiler(get_kernels_s)

    def compile_nonsinglet(self):
        """Compiles the non-singlet integration kernel for each basis """
        return self._compiler(get_kernel_ns)

    def njit(self, function):
        """
            Do nb.njit to the function if the `numba_it` flag is set to True.

            Check whether a list of functions is passed.

            Parameters
            ---------
                function : function
                    input function

            Returns
            -------
                funciton : function
                    compiled function
        """
        if isinstance(function, abc.Iterable):
            return [self.njit(f) for f in function]
        if self.numba_it:
            return nb.njit(function)
        else:
            return function
