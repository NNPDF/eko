"""The unpolarized, time-like |NLO| matching conditions."""

import numba as nb
import numpy as np

from eko.constants import CF


@nb.njit(cache=True)
def A_hg(N, L):
    r"""Compute the |NLO| heavy-gluon |OME|.

    Implements :eqref:`27` from :cite:`Cacciari:2005ry`.

    Parameters
    ----------
    N : complex
        Mellin moment
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    A_hg : complex
        |NLO| heavy-gluon |OME|
        :math:`A_{hg}^{S,(1)}`
    """
    result = (
        2
        * CF
        * (
            (2 + N + N**2) / (N * (N**2 - 1)) * (L - 1)
            + 4 / (N - 1) ** 2
            - 4 / N**2
            + 2 / (N + 1) ** 2
        )
    )
    return result


@nb.njit(cache=True)
def A_gg(L):
    r"""Compute the |NLO| gluon-gluon |OME|.

    Implements the Mellin transform of
    :eqref:`24` from :cite:`Cacciari:2005ry`.
    It is identical to the one in
    :class:`~ekore.operator_matrix_elements.unpolarized.space_like.as1`.

    Parameters
    ----------
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    A_gg : complex
        |NLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(1)}`
    """
    return -2.0 / 3.0 * L


@nb.njit(cache=True)
def A_singlet(N, L):
    r"""Compute the |NLO| singlet |OME|.

    Parameters
    ----------
    N : complex
        Mellin moment
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    numpy.ndarray
        |NLO| singlet |OME|
        :math:`A^{S,(1)}`
    """
    result = np.array(
        [[A_gg(L), 0 + 0j, 0], [0 + 0j, 0, 0], [A_hg(N, L), 0, 0]], np.complex128
    )
    return result


@nb.njit(cache=True)
def A_ns():
    r"""Compute the |NLO| non-singlet |OME|.

    Returns
    -------
    A_ns : numpy.ndarray
        |NLO| non-singlet |OME|
        :math:`A^{S,(1)}`
    """
    result = np.array([[0, 0], [0, 0]], np.complex128)
    return result
