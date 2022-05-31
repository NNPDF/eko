# -*- coding: utf-8 -*-

import numba as nb
import numpy as np

from . import interpolation, mellin

spec = [
    ("ker_mode", nb.int),
    ("u", nb.float64),
    ("is_log", nb.boolean),
    ("logx", nb.float64),
    ("areas", nb.float64[:, :]),
    ("mode0", nb.int),
    ("mode1", nb.int),
]


@nb.experimental.jitclass(spec)
class QuadKerBase:
    """Manage the common part of Mellin inversion integral.

    Parameters
    ----------
    ker_mode : int
        kernel mode: 0 for anomalous dimension, 1 for matching
    u : float
        quad argument
    is_log : boolean
        is a logarithmic interpolation
    logx : float
        Mellin inversion point
    areas : tuple
        basis function configuration
    mode0 : str
        first sector element
    mode1 : str
        second sector element
    """

    def __init__(self, ker_mode, u, is_log, logx, areas, mode0, mode1):
        self.ker_mode = ker_mode
        self.u = u
        self.is_log = is_log
        self.logx = logx
        self.areas = areas
        self.mode0 = mode0
        self.mode1 = mode1

    @property
    def path(self):
        """Returns the associated instance of :class:`eko.mellin.Path`"""
        return mellin.Path(self.u, self.logx, self.is_singlet)

    @property
    def n(self):
        """Returns the Mellin moment N"""
        return self.path.n

    @property
    def is_singlet(self):
        """Returns the Mellin moment N"""
        return self.mode0 in [100, 21, 90]

    def select_element(self, ker):
        """Select correct element of (matrix-valued) kernel.

        Parameters
        ----------
        ker : numpy.complex or numpy.ndarray
            integration kernel
        mode0 : int
            id for first sector element
        mode1 : int
            id for second sector element

        Returns
        -------
        ker : complex
            singlet integration kernel element
        """
        # OME mode
        if self.ker_mode == 1:
            if self.is_singlet:
                indices = {21: 0, 100: 1, 90: 2}
            else:
                indices = {200: 0, 91: 1}
            return ker[indices[self.mode0], indices[self.mode1]]
        # AD mode
        k = 0 if self.mode0 == 100 else 1
        l = 0 if self.mode1 == 100 else 1
        return ker[k, l]

    @property
    def integrand(self):
        """Get transformation to Mellin space integral.

        Returns
        -------
        complex
            common mellin inversion intgrand
        """
        if self.logx == 0.0:
            return 0.0
        pj = interpolation.evaluate_grid(
            self.path.n, self.is_log, self.logx, self.areas
        )
        if pj == 0.0:
            return 0.0
        return self.path.prefactor * pj * self.path.jac

    @property
    def is_empty(self):
        """Is the kernel exactly 0?

        Returns
        -------
        bool
            Is kernel contributing?
        """
        return self.integrand == 0.0

    def compute_matching(self, ker):
        """Compute the matching kernel.

        Parameters
        ----------
        ker : float or np.ndarray
            matching kernel

        Returns
        -------
        float
            actual mellin kernel
        """
        ker = self.select_element(ker)
        return np.real(ker * self.integrand)
