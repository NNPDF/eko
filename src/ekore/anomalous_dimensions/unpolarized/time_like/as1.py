"""
The unpolarized time-like leading-order 
(LO) Altarelli-Parisi splitting kernels.

"""

import numba as nb
import numpy as np
from eko import constants



@nb.njit(cache=True)
def gamma_qq(N, s1):
    """Computes the LO quark-quark anomalous dimension.
    Implements Eqn. (B.3) from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment 
    s1 : complex
        Harmonic sum $S_{1}$ 

    Returns
    -------
    gamma_qq : complex
        LO quark-quark anomalous dimension 
        :math:`\gamma_{qq}^{(0)}(N)` 

    """
    return constants.CF * (-3.0 + (4.0 * s1) - 2.0 / (N * (N + 1.0)))

@nb.njit(cache=True)
def gamma_qg(N):
    """Computes the LO quark-gluon anomalous dimension.
    Implements Eqn. (B.4) from :cite:`Mitov:2006wy` 
    and Eqn. (A1) from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment 

    Returns
    -------
    gamma_qg : complex
        LO quark-gluon anomalous dimension 
        :math:`\gamma_{qg}^{(0)}(N)` 

    """
    return - (N**2 + N + 2.0) / (N * (N + 1.0) * (N + 2.0))

@nb.njit(cache=True)
def gamma_gq(N, nf):
    """Computes the LO gluon-quark anomalous dimension.
    Implements Eqn. (B.5) from :cite:`Mitov:2006wy` 
    and Eqn. (A1) from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment 
    nf : int
        No. of active flavors 

    Returns
    -------
    gamma_qg : complex
        LO quark-gluon anomalous dimension 
        :math:`\gamma_{gq}^{(0)}(N)`

    """
    return -4.0 * nf * constants.CF * (N**2 + N + 2.0) / (N * (N - 1.0) * (N + 1.0))

@nb.njit(cache=True)
def gamma_gg(N, s1, nf):
    """Computes the LO gluon-gluon anomalous dimension.
    Implements Eqn. (B.6) from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment 
    s1 : complex
        Harmonic sum $S_{1}$ 
    nf : int
        No. of active flavors 

    Returns
    -------
    gamma_qq : complex
        LO quark-quark anomalous dimension 
        :math:`\gamma_{gg}^{(0)}(N)`

    """
    return ((2.0 * nf - 11.0 * constants.CA) / 3.0 + 4.0 * constants.CA 
    * (s1 - 1.0 / (N * (N - 1.0)) - 1.0 / ((N + 1.0) * (N + 2.0))))

@nb.njit(cache=True)
def gamma_ns(N, s1):
    """Computes the LO non-singlet anomalous dimension.
    At LO, :math:`\gamma_{ns}^{(0)} = \gamma_{qq}^{(0)}`.
	
    Parameters
    ----------
    N : complex
        Mellin moment 
    s1 : complex
        Harmonic sum $S_{1}$ 

    Returns
    -------
    gamma_ns : complex
        LO quark-quark anomalous dimension 
        :math:`\gamma_{ns}^{(0)}(N)`

    """
    return gamma_qq(N, s1)

@nb.njit(cache=True)
def gamma_singlet(N, s1, nf):
    """Computes the LO singlet anomalous dimension matrix.
    Implements Eqn. (2.13) from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment 
    s1 : complex
        Harmonic sum $S_{1}$ 
    nf : int
        No. of active flavors 

    Returns
    -------
    gamma_singlet : numpy.ndarray
        LO singlet anomalous dimension matrix 
        :math:`\gamma_{s}^{(0)}`

    """
    return np.array([[gamma_qq(N, s1), gamma_gq(N, nf)], 
    [gamma_qg(N), gamma_gg(N, s1, nf)]], np.complex_)