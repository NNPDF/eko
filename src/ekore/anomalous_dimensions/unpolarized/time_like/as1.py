"""The unpolarized, time-like |LO| Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from eko import constants

from ....harmonics import cache as c


@nb.njit(cache=True)
def gamma_qq(N, cache):
    r"""Compute the |LO| quark-quark anomalous dimension.

    Implements :eqref:`B.3` from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        LO quark-quark anomalous dimension
        :math:`\gamma_{qq}^{(0)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    result = constants.CF * (-3.0 + (4.0 * S1) - 2.0 / (N * (N + 1.0)))
    return result


@nb.njit(cache=True)
def gamma_qg(N):
    r"""Compute the |LO| quark-gluon anomalous dimension.

    Implements :eqref:`B.4` from :cite:`Mitov:2006wy`
    and :eqref:`A1` from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment

    Returns
    -------
    complex
        LO quark-gluon anomalous dimension
        :math:`\gamma_{qg}^{(0)}(N)`
    """
    result = -(N**2 + N + 2.0) / (N * (N + 1.0) * (N + 2.0))
    return result


@nb.njit(cache=True)
def gamma_gq(N, nf):
    r"""Compute the |LO| gluon-quark anomalous dimension.

    Implements :eqref:`B.5` from :cite:`Mitov:2006wy`
    and :eqref:`A1` from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors

    Returns
    -------
    complex
        LO quark-gluon anomalous dimension
        :math:`\gamma_{gq}^{(0)}(N)`
    """
    result = -4.0 * nf * constants.CF * (N**2 + N + 2.0) / (N * (N - 1.0) * (N + 1.0))
    return result


@nb.njit(cache=True)
def gamma_gg(N, nf, cache):
    r"""Compute the |LO| gluon-gluon anomalous dimension.

    Implements :eqref:`B.6` from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        LO quark-quark anomalous dimension
        :math:`\gamma_{gg}^{(0)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    result = (2.0 * nf - 11.0 * constants.CA) / 3.0 + 4.0 * constants.CA * (
        S1 - 1.0 / (N * (N - 1.0)) - 1.0 / ((N + 1.0) * (N + 2.0))
    )
    return result


@nb.njit(cache=True)
def gamma_ns(N, cache):
    r"""Compute the |LO| non-singlet anomalous dimension.

    At LO, :math:`\gamma_{ns}^{(0)} = \gamma_{qq}^{(0)}`.

    Parameters
    ----------
    N : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        LO quark-quark anomalous dimension
        :math:`\gamma_{ns}^{(0)}(N)`
    """
    return gamma_qq(N, cache)


@nb.njit(cache=True)
def gamma_singlet(N, nf, cache):
    r"""Compute the |LO| singlet anomalous dimension matrix.

    Implements :eqref:`2.13` from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    numpy.ndarray
        LO singlet anomalous dimension matrix
        :math:`\gamma_{s}^{(0)}`
    """
    result = np.array(
        [
            [gamma_qq(N, cache), gamma_gq(N, nf)],
            [gamma_qg(N), gamma_gg(N, nf, cache)],
        ],
        np.complex128,
    )
    return result
