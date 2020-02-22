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
import numba as nb

import eko.alpha_s as alpha_s
import eko.splitting_functions_LO as sf_LO
import eko.mellin as mellin

import logging
logger = logging.getLogger(__name__)

def get_kernel_ns(basis_function, nf, constants, beta_0):
    """Returns the non-singlet integration kernel"""
    CA = constants.CA
    CF = constants.CF

    def ker(n, lnx, delta_t):
        """true non-siglet integration kernel"""
        ln = -delta_t * sf_LO.gamma_ns_0(n, nf, CA, CF) / beta_0
        interpoln = basis_function(n, lnx)
        return np.exp(ln) * interpoln

    return ker


def get_kernels_s(basis_function, nf, constants, beta_0):
    """Return all singlet integration kernels"""
    CA = constants.CA
    CF = constants.CF

    def get_ker(k, l):
        """true singlet kernel"""

        def ker(N, lnx, delta_t):
            """a singlet integration kernel"""
            l_p, l_m, e_p, e_m = sf_LO.get_Eigensystem_gamma_singlet_0(N, nf, CA, CF)
            ln_p = -delta_t * l_p / beta_0
            ln_m = -delta_t * l_m / beta_0
            interpoln = basis_function(N, lnx)
            return (e_p[k][l] * np.exp(ln_p) + e_m[k][l] * np.exp(ln_m)) * interpoln

        return ker

    return get_ker(0, 0), get_ker(0, 1), get_ker(1, 0), get_ker(1, 1)

def prepare_singlet(kernels, path, jac):
    """ Return a list of integrands prepare to be run """
    integrands = []
    logger.info("Singlet operator: kernel compilation started")
    for kernel_set in kernels:
        kernel_int = []
        for ker in kernel_set:
            kernel_int.append(mellin.compile_integrand(ker, path, jac))
        integrands.append(kernel_int)
    logger.info("Singlet operator: kernel compilation finished")
    return integrands

def prepare_non_singlet(kernels, path, jac):
    """ Return a list of integrands prepare to be run """
    integrands = []
    logger.info("Non-singlet operator: kernel compilation started")
    for ker in kernels:
        kernel_int = mellin.compile_integrand(ker, path, jac)
        integrands.append(kernel_int)
    logger.info("Non-singlet operator: kernel compilation finished")
    return integrands


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

    def __init__(self, interpol_dispatcher, constants, delta_t = None, numba_it=True):
        self.interpol_dispatcher = interpol_dispatcher
        self.constants = constants
        self.numba_it = numba_it
        self.integrands_ns = {}
        self.integrands_s = {}
        self.delta_t = delta_t

    def _compiler(self, generating_function, nf):
        """
            Call `generating_function` with the appropiate parameters
            and pass the output through numba
        """
        beta_0 = alpha_s.beta_0(nf, self.constants.CA, self.constants.CF, self.constants.TF)
        kernels = []
        for basis_function in self.interpol_dispatcher:
            new_ker = generating_function(
                basis_function.callable,
                nf,
                self.constants,
                beta_0,
            )
            kernels.append(self.njit(new_ker))
        return kernels

    def compile_singlet(self, nf):
        """Compiles the singlet integration kernels for each basis """
        ker = self._compiler(get_kernels_s, nf)
        return ker

    def compile_nonsinglet(self, nf):
        """Compiles the non-singlet integration kernel for each basis """
        ker = self._compiler(get_kernel_ns, nf)
        return ker

    def set_up_all_integrands(self, nf_values):
        """ Compiles singlet and non-singlet integration kernel for each basis """
        if isinstance(nf_values, (np.int, np.integer)):
            nf_values = [nf_values]
        # Setup path
        path, jac = mellin.get_path_Talbot()
        for nf in nf_values:
            if nf not in self.integrands_s:
                self.integrands_s[nf] = prepare_singlet(self.compile_singlet(nf), path, jac)
            if nf not in self.integrands_ns:
                self.integrands_ns[nf] = prepare_non_singlet(self.compile_nonsinglet(nf), path, jac)

    def get_singlet_for_nf(self, nf):
        integrands = self.integrands_s.get(nf)
        if integrands is None:
            self.set_up_all_integrands([nf])
            integrands = self.integrands_s[nf]
        return integrands

    def get_non_singlet_for_nf(self, nf):
        integrands = self.integrands_ns.get(nf)
        if integrands is None:
            self.set_up_all_integrands([nf])
            integrands = self.integrands_ns[nf]
        return integrands

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
