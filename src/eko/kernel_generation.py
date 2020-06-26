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


def get_kernel_ns(basis_function, nf, constants):
    r"""
        Returns the non-singlet integration kernel
        :math:`\tilde E_{ns}(t_0 \leftarrow t_1)`.

        Parameters
        ----------
            basis_function : callable
                accompainging basis function
            nf : int
                number of active flavors
            constants : eko.constants.Constants
                active configuration

        Returns
        -------
            ker : callable
                (physical) kernel, which will be further modified for the
                actual Mellin implementation
    """
    CA = constants.CA
    CF = constants.CF
    beta_0 = sc.beta_0(nf, constants.CA, constants.CF, constants.TF)

    def ker(n, lnx, delta_t):
        """true non-siglet integration kernel"""
        ln = -delta_t * ad_lo.gamma_ns_0(n, nf, CA, CF) / beta_0
        interpoln = basis_function(n, lnx)
        return np.exp(ln) * interpoln

    return ker


def get_kernels_s(basis_function, nf, constants):
    r"""
        Returns the singlet integration kernels
        :math:`\ES{t_0}{t_1}`.

        Parameters
        ----------
            basis_function : callable
                accompainging basis function
            nf : int
                number of active flavors
            constants : eko.constants.Constants
                active configuration

        Returns
        -------
            ker : list(callable)
                (physical) kernels, which will be further modified for the
                actual Mellin implementation
    """
    CA = constants.CA
    CF = constants.CF
    beta_0 = sc.beta_0(nf, constants.CA, constants.CF, constants.TF)

    def get_ker(k, l):
        """(k,l)-th element of singlet kernel matrix"""

        def ker(N, lnx, delta_t):
            """a singlet integration kernel"""
            l_p, l_m, e_p, e_m = ad_lo.get_Eigensystem_gamma_singlet_0(N, nf, CA, CF)
            ln_p = -delta_t * l_p / beta_0
            ln_m = -delta_t * l_m / beta_0
            interpoln = basis_function(N, lnx)
            return (e_p[k][l] * np.exp(ln_p) + e_m[k][l] * np.exp(ln_m)) * interpoln

        return ker

    return get_ker(0, 0), get_ker(0, 1), get_ker(1, 0), get_ker(1, 1)


def prepare_singlet(kernels, path, jac):
    """ Return a list of integrands prepare to be run """
    integrands = []
    for kernel_set in kernels:
        kernel_int = []
        for ker in kernel_set:
            kernel_int.append(mellin.compile_integrand(ker, path, jac))
        integrands.append(kernel_int)
    return integrands


def prepare_non_singlet(kernels, path, jac):
    """ Return a list of integrands prepare to be run """
    integrands = []
    for ker in kernels:
        kernel_int = mellin.compile_integrand(ker, path, jac)
        integrands.append(kernel_int)
    return integrands


class KernelDispatcher:
    """
        The kernel dispatcher does the common preparation for the kernel functions

        Upon calling the appropiate `compile_*` method the dispatcher will return
        a `numba` compiled kernel.

        Parameters
        ----------
            interpol_dispatcher : InterpolatorDispatcher
                An instance of the InterpolatorDispatcher class
            constants : Constants
                An instance of the Constants class
            nf : float
                Number of flavour to consider
            numba_it : bool  (default: True)
                If true, the functions will be `numba` compiled
    """

    def __init__(self, interpol_dispatcher, constants, numba_it=True):
        self.interpol_dispatcher = interpol_dispatcher
        self.constants = constants
        self.numba_it = numba_it
        self.integrands_ns = {}
        self.integrands_s = {}

    def _compiler(self, generating_function, nf):
        """
            Iterate `generating_function` alogn the basis functions, call with
            the appropiate parameters and pass the output through numba.

            Parameters
            ----------
                generating_function : callable
                    getter for the actual kernel
                nf : int
                    number of active flavors

            Returns
            -------
                kernels : list(callable)
                    list of basis functions combined with input function
        """
        kernels = []
        for basis_function in self.interpol_dispatcher:
            new_ker = generating_function(basis_function.callable, nf, self.constants)
            kernels.append(self.njit(new_ker))
        return kernels

    def set_up_all_integrands(self, nf_values):
        """
            Compiles singlet and non-singlet integration kernel for each basis function.

            Parameters
            ----------
                nf_values : int or list(int)
                    (list of) number of active flavors
        """
        if isinstance(nf_values, (np.int, np.integer)):
            nf_values = [nf_values]
        # Setup path
        path, jac = mellin.get_path_Talbot()
        for nf in nf_values:
            if nf not in self.integrands_s:
                self.integrands_s[nf] = prepare_singlet(
                    self._compiler(get_kernels_s, nf), path, jac
                )
            if nf not in self.integrands_ns:
                self.integrands_ns[nf] = prepare_non_singlet(
                    self._compiler(get_kernel_ns, nf), path, jac
                )

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
