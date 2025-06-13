r"""The unpolarized, spacelike |NLO| |OME|.

Heavy quark contribution for intrinsic evolution are taken from :cite:`Ball_2016`
and Mellin transformed with Mathematica.
The other matching conditions for the |VFNS| at :math:`\mu_F^2 \neq m_H^2`
are provided in :cite:`Buza_1998`.
"""

import numba as nb
import numpy as np

from eko.constants import CF

from ....harmonics import cache as c


@nb.njit(cache=True)
def A_hh(n, cache, L):
    r"""|NLO| heavy-heavy |OME| :math:`A_{HH}^{(1)}`.

    They are defined as the Mellin transform of :math:`K_{hh}`
    given in :eqref:`20a` of :cite:`Ball_2016`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    complex
        |NLO| heavy-heavy |OME| :math:`A_{HH}^{(1)}`
    """
    S1m = c.get(c.S1, cache, n) - 1 / n  # harmonics.S1(n - 1)
    S2m = c.get(c.S2, cache, n) - 1 / n**2  # harmonics.S2(n - 1)
    ahh_l = (2 + n - 3 * n**2) / (n * (1 + n)) + 4 * S1m
    ahh = 2 * (
        2 + 5 * n + n**2 - 6 * n**3 - 2 * n**4 - 2 * n * (-1 - 2 * n + n**3) * S1m
    ) / (n * (1 + n)) ** 2 + 4 * (S1m**2 + S2m)
    return -CF * (ahh_l * L + ahh)


@nb.njit(cache=True)
def A_gh(n, L):
    r"""|NLO| gluon-heavy |OME| :math:`A_{gH}^{(1)}`.

    They are defined as the Mellin transform of :math:`K_{gh}`
    given in :eqref:`20b` of :cite:`Ball_2016`.

    Parameters
    ----------
    n : complex
        Mellin moment
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    complex
        |NLO| gluon-heavy |OME| :math:`A_{gH}^{(1)}`
    """
    agh_l1 = (2 + n + n**2) / (n * (n**2 - 1))
    agh_l0 = (-4 + 2 * n + n**2 * (15 + n * (3 + n - n**2))) / (n * (n**2 - 1)) ** 2
    return 2 * CF * (agh_l0 + agh_l1 * L)


@nb.njit(cache=True)
def A_hg(n, L):
    r"""|NLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(1)}`.

    They are defined as the Mellin transform of:eqref:`B.2` from :cite:`Buza_1998`.

    Parameters
    ----------
    n : complex
        Mellin moment
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    complex
        |NLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(1)}`
    """
    den = 1.0 / (n * (n + 1) * (2 + n))
    num = 2 * (2 + n + n**2)
    return num * den * L


@nb.njit(cache=True)
def A_gg(L):
    r"""|NLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(1)}`.

    They are defined as the Mellin transform of :eqref:`B.6` from :cite:`Buza_1998`.

    Parameters
    ----------
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    complex
        |NLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(1)}`
    """
    return -2.0 / 3.0 * L


@nb.njit(cache=True)
def A_singlet(n, cache, L):
    r"""Compute the |NLO| singlet |OME|.

    .. math::
        A^{S,(1)} = \left(\begin{array}{cc}
        A_{gg,H}^{S,(1)} & 0  & A_{gH}^{(1)} \\
        0 & 0 & 0 \\
        A_{hg}^{S,(1)} & 0 & A_{HH}^{(1)}
        \end{array}\right)

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    numpy.ndarray
        |NLO| singlet |OME| :math:`A^{S,(1)}`
    """
    A_S = np.array(
        [
            [A_gg(L), 0.0, A_gh(n, L)],
            [0 + 0j, 0 + 0j, 0 + 0j],
            [A_hg(n, L), 0.0, A_hh(n, cache, L)],
        ],
        np.complex128,
    )
    return A_S


@nb.njit(cache=True)
def A_ns(n, cache, L):
    r"""Compute the |NLO| non-singlet |OME| with intrinsic contributions.

    .. math::
        A^{NS,(1)} = \left(\begin{array}{cc}
        0 & 0 \\
        0 & A_{HH}^{(1)}
        \end{array}\right)

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    numpy.ndarray
        |NLO| non-singlet |OME| :math:`A^{S,(1)}`
    """
    return np.array([[0 + 0j, 0 + 0j], [0 + 0j, A_hh(n, cache, L)]], np.complex128)
