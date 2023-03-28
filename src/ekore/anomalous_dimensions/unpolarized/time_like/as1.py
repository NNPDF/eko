"""The unpolarized |LO| Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from eko import constants

from ....harmonics import w1


@nb.njit(cache=True)
def gamma_qq(N):
    r"""Compute the LO quark-quark anomalous dimension.

    Implements :eqref:`B.3` from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment

    Returns
    -------
    gamma_qq : complex
        LO quark-quark anomalous dimension
        :math:`\gamma_{qq}^{(0)}(N)`

    """
    s1 = w1.S1(N)
    result = constants.CF * (-3.0 + (4.0 * s1) - 2.0 / (N * (N + 1.0)))
    return result


@nb.njit(cache=True)
def gamma_qg(N):
    r"""Compute the LO quark-gluon anomalous dimension.

    Implements :eqref:`B.4` from :cite:`Mitov:2006wy`
    and :eqref:`A1` from :cite:`Gluck:1992zx`.

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
    result = -(N**2 + N + 2.0) / (N * (N + 1.0) * (N + 2.0))
    return result


@nb.njit(cache=True)
def gamma_gq(N, nf):
    r"""Compute the LO gluon-quark anomalous dimension.

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
    gamma_qg : complex
        LO quark-gluon anomalous dimension
        :math:`\gamma_{gq}^{(0)}(N)`

    """
    result = -4.0 * nf * constants.CF * (N**2 + N + 2.0) / (N * (N - 1.0) * (N + 1.0))
    return result


@nb.njit(cache=True)
def gamma_gg(N, nf):
    r"""Compute the LO gluon-gluon anomalous dimension.

    Implements :eqref:`B.6` from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors

    Returns
    -------
    gamma_qq : complex
        LO quark-quark anomalous dimension
        :math:`\gamma_{gg}^{(0)}(N)`

    """
    s1 = w1.S1(N)
    result = (2.0 * nf - 11.0 * constants.CA) / 3.0 + 4.0 * constants.CA * (
        s1 - 1.0 / (N * (N - 1.0)) - 1.0 / ((N + 1.0) * (N + 2.0))
    )
    return result


@nb.njit(cache=True)
def gamma_ns(N):
    r"""Compute the LO non-singlet anomalous dimension.

    At LO, :math:`\gamma_{ns}^{(0)} = \gamma_{qq}^{(0)}`.

    Parameters
    ----------
    N : complex
        Mellin moment

    Returns
    -------
    gamma_ns : complex
        LO quark-quark anomalous dimension
        :math:`\gamma_{ns}^{(0)}(N)`

    """
    return gamma_qq(N)


@nb.njit(cache=True)
def gamma_singlet(N, nf):
    r"""Compute the LO singlet anomalous dimension matrix.

    Implements :eqref:`2.13` from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors

    Returns
    -------
    gamma_singlet : numpy.ndarray
        LO singlet anomalous dimension matrix
        :math:`\gamma_{s}^{(0)}`
    """
    result = np.array(
        [
            [gamma_qq(N), gamma_gq(N, nf)],
            [gamma_qg(N), gamma_gg(N, nf)],
        ],
        np.complex_,
    )
    return result
