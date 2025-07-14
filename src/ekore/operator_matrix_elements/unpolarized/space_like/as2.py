r"""The unpolarized, spacelike |NNLO| |OME|.

See, :cite:`Buza_1998` appendix B.
The expression for :math:`\mu_F^2 = m_H^2` are taken from :cite:`Vogt:2004ns` directly in N space.
While the parts proportional to :math:`\ln(\mu_F^2 / m_h^2)` comes |QCDNUM|
(https://github.com/N3PDF/external/blob/master/qcdnum/qcdnum/pij/ome.f)
and Mellin transformed with Mathematica.

The expression for ``A_Hg_l0`` comes form :cite:`Bierenbaum:2009zt`.
"""

import numba as nb
import numpy as np

from eko import constants
from eko.constants import zeta2, zeta3

from ....harmonics import cache as c
from .as1 import A_gg as A_gg_1
from .as1 import A_hg as A_hg_1


@nb.njit(cache=True)
def A_qq_ns(n, cache, L):
    r"""|NNLO| light-light non-singlet |OME| :math:`A_{qq,H}^{NS,(2)}`.

    It is given in :eqref:`B.4` of :cite:`Buza_1998`.

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
        |NNLO| light-light non-singlet |OME| :math:`A_{qq,H}^{NS,(2)}`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S1m = S1 - 1 / n  # harmonic_S1(n - 1)
    S2m = S2 - 1 / n**2  # harmonic_S2(n - 1)

    a_qq_l0 = (
        -224.0 / 27.0 * (S1 - 1.0 / n)
        - 8.0 / 3.0 * zeta3
        + 40 / 9.0 * zeta2
        + 73.0 / 18.0
        + 44.0 / 27.0 / n
        - 268.0 / 27.0 / (n + 1.0)
        + 8.0 / 3.0 * (-1.0 / n**2 + 1.0 / (n + 1.0) ** 2)
        + 20.0 / 9.0 * (S2 - 1.0 / n**2 - zeta2 + S2 + 1.0 / (n + 1.0) ** 2 - zeta2)
        + 2.0
        / 3.0
        * (-2.0 * (S3 - 1.0 / n**3 - zeta3) - 2.0 * (S3 + 1.0 / (n + 1.0) ** 3 - zeta3))
    )

    a_qq_l1 = (
        2 * (-12 - 28 * n + 9 * n**2 + 34 * n**3 - 3 * n**4) / (9 * (n * (n + 1)) ** 2)
        + 80 / 9 * S1m
        - 16 / 3 * S2m
    )

    a_qq_l2 = -2 * ((2 + n - 3 * n**2) / (3 * n * (n + 1)) + 4 / 3 * S1m)
    return constants.CF * constants.TR * (a_qq_l2 * L**2 + a_qq_l1 * L + a_qq_l0)


@nb.njit(cache=True)
def A_hq_ps(n, cache, L):
    r"""|NNLO| heavy-light pure-singlet |OME| :math:`A_{Hq}^{PS,(2)}`.

    It is given in :eqref:`B.1` of :cite:`Buza_1998`.

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
        |NNLO| heavy-light pure-singlet |OME| :math:`A_{Hq}^{PS,(2)}`
    """
    S2 = c.get(c.S2, cache, n)

    F1M = 1.0 / (n - 1.0) * (zeta2 - (S2 - 1.0 / n**2))
    F11 = 1.0 / (n + 1.0) * (zeta2 - (S2 + 1.0 / (n + 1.0) ** 2))
    F12 = 1.0 / (n + 2.0) * (zeta2 - (S2 + 1.0 / (n + 1.0) ** 2 + 1.0 / (n + 2.0) ** 2))
    F21 = -F11 / (n + 1.0)
    a_hq_l0 = (
        -(
            32.0 / 3.0 / (n - 1.0)
            + 8.0 * (1.0 / n - 1.0 / (n + 1.0))
            - 32.0 / 3.0 * 1.0 / (n + 2.0)
        )
        * zeta2
        - 448.0 / 27.0 / (n - 1.0)
        - 4.0 / 3.0 / n
        - 124.0 / 3.0 * 1.0 / (n + 1.0)
        + 1600.0 / 27.0 / (n + 2.0)
        - 4.0 / 3.0 * (-6.0 / n**4 - 6.0 / (n + 1.0) ** 4)
        + 2.0 * 2.0 / n**3
        + 10.0 * 2.0 / (n + 1.0) ** 3
        + 16.0 / 3.0 * 2.0 / (n + 2.0) ** 3
        - 16.0 * zeta2 * (-1.0 / n**2 - 1.0 / (n + 1.0) ** 2)
        + 56.0 / 3.0 / n**2
        + 88.0 / 3.0 / (n + 1.0) ** 2
        + 448.0 / 9.0 / (n + 2.0) ** 2
        + 32.0 / 3.0 * F1M
        + 8.0 * ((zeta2 - S2) / n - F11)
        - 32.0 / 3.0 * F12
        + 16.0 * (-(zeta2 - S2) / n**2 + F21)
    )

    a_hq_l1 = (
        8
        * (2 + n * (5 + n))
        * (4 + n * (4 + n * (7 + 5 * n)))
        / ((n - 1) * (n + 2) ** 2 * (n * (n + 1)) ** 3)
    )

    a_hq_l2 = -4 * (2 + n + n**2) ** 2 / ((n - 1) * (n + 2) * (n * (n + 1)) ** 2)

    return constants.CF * constants.TR * (a_hq_l2 * L**2 + a_hq_l1 * L + a_hq_l0)


@nb.njit(cache=True)
def A_hg(n, cache, L):
    r"""|NNLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(2)}`.

    It is given in :eqref:`B.3` of :cite:`Buza_1998`.
    The expession for ``A_Hg_l0`` comes form :cite:`Bierenbaum:2009zt`.

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
        |NNLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(2)}`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    Sm2 = c.get(c.Sm2, cache, n, is_singlet=True)
    S3 = c.get(c.S3, cache, n)
    Sm21 = c.get(c.Sm21, cache, n, is_singlet=True)
    Sm3 = c.get(c.Sm3, cache, n, is_singlet=True)
    S1m = S1 - 1 / n
    S2m = S2 - 1 / n**2

    a_hg_l0 = (
        -(
            3084
            + 192 / n**4
            + 1056 / n**3
            + 2496 / n**2
            + 2928 / n
            + 2970 * n
            + 1782 * n**2
            + 6 * n**3
            - 1194 * n**4
            - 1152 * n**5
            - 516 * n**6
            - 120 * n**7
            - 12 * n**8
        )
        / ((n - 1) * ((1 + n) * (2 + n)) ** 4)
        + (
            764
            - 16 / n**4
            - 80 / n**3
            - 100 / n**2
            + 3 * 72 / n
            + 208 * n**3
            + 3 * (288 * n + 176 * n**2 + 16 * n**4)
        )
        / (3 * (1 + n) ** 4 * (2 + n))
        + 12 * Sm3 * (2 + n + n**2) / (n * (1 + n) * (2 + n))
        - 24 * Sm2 * (4 + n - n**2) / ((1 + n) * (2 + n)) ** 2
        - S1
        * (48 / n + 432 + 564 * n + 324 * n**2 + 138 * n**3 + 48 * n**4 + 6 * n**5)
        / ((1 + n) * (2 + n)) ** 3
        + S1
        * (-160 - 32 / n**2 - 80 / n + 8 * n * (n - 1))
        / (3 * (1 + n) ** 2 * (2 + n))
        - 6 * S1**2 * (11 + 8 * n + n**2 + 2 / n) / ((1 + n) * (2 + n)) ** 2
        + 8 * S1**2 * (2 / (3 * n) + 1) / (n * (2 + n))
        - 2
        * S2
        * (63 + 48 / n**2 + 54 / n + 39 * n + 63 * n**2 + 21 * n**3)
        / ((n - 1) * (1 + n) ** 2 * (2 + n) ** 2)
        + 8 * S2 * (17 - 2 / n**2 - 5 / n + n * (17 + n)) / (3 * (1 + n) ** 2 * (2 + n))
        + (1 + 2 / n + n)
        / ((1 + n) * (2 + n))
        * (24 * Sm2 * S1 + 10 * S1**3 / 9 + 46 * S1 * S2 / 3 + 176 * S3 / 9 - 24 * Sm21)
    )

    a_hg_l1 = (
        2
        * (
            +640
            + 2192 * n
            + 2072 * n**2
            + 868 * n**3
            + 518 * n**4
            + 736 * n**5
            + 806 * n**6
            + 542 * n**7
            + 228 * n**8
            + 38 * n**9
        )
        / (3 * (n * (n + 1) * (n + 2)) ** 3 * (n - 1))
    )

    a_hg_l1 -= (
        2
        * (
            n
            * (n**2 - 1)
            * (n + 2)
            * (
                4 * (36 + n * (88 + n * (33 + n * (8 + 9 * n)))) * S1m
                + n
                * (n + 1)
                * (n + 2)
                * (2 + n + n**2)
                * (10 * S1m**2 + 18 * (2 * Sm2 + zeta2) + 26 * S2m)
            )
        )
        / (3 * (n * (n + 1) * (n + 2)) ** 3 * (n - 1))
    )
    a_hg_l1 += 12 * zeta2 * (-2 + n + n**3) / (n * (n**2 - 1) * (n + 2))

    a_hg_l2 = (
        4
        * (2 + n + n**2)
        * (2 * (-11 + n + n**2) * (1 + n + n**2) / (n - 1))
        / (3 * (n * (n + 1) * (n + 2)) ** 2)
    ) + 20 * (2 + n + n**2) * S1 / (3 * n * (n + 1) * (n + 2))

    return a_hg_l2 * L**2 + a_hg_l1 * L + a_hg_l0


@nb.njit(cache=True)
def A_gq(n, cache, L):
    r"""|NNLO| gluon-quark |OME| :math:`A_{gq,H}^{S,(2)}`.

    It is given in :eqref:`B.5` of :cite:`Buza_1998`.

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
        |NNLO| gluon-quark |OME| :math:`A_{gq,H}^{S,(2)}`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S1m = S1 - 1 / n  # harmonic_S1(n - 1)

    B2M = ((S1 - 1.0 / n) ** 2 + S2 - 1.0 / n**2) / (n - 1.0)
    B21 = ((S1 + 1.0 / (n + 1.0)) ** 2 + S2 + 1.0 / (n + 1.0) ** 2) / (n + 1.0)

    a_gq_l0 = (
        4.0 / 3.0 * (2.0 * B2M - 2.0 * (S1**2 + S2) / n + B21)
        + 8.0
        / 9.0
        * (
            -10.0 * (S1 - 1.0 / n) / (n - 1.0)
            + 10.0 * S1 / n
            - 8.0 * (S1 + 1.0 / (n + 1.0)) / (n + 1.0)
        )
        + 1.0 / 27.0 * (448.0 * (1.0 / (n - 1.0) - 1.0 / n) + 344.0 / (n + 1.0))
    )

    a_gq_l1 = -(
        -96
        + 16 * n * (7 + n * (21 + 10 * n + 8 * n**2))
        - 48 * n * (1 + n) * (2 + n + n**2) * S1m
    ) / (9 * (n - 1) * (n * (1 + n)) ** 2)

    a_gq_l2 = 8 * (2 + n + n**2) / (3 * n * (n**2 - 1))

    return constants.CF * constants.TR * (a_gq_l2 * L**2 + a_gq_l1 * L + a_gq_l0)


@nb.njit(cache=True)
def A_gg(n, cache, L):
    r"""|NNLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(2)}`.

    It is given in :eqref:`B.7` of :cite:`Buza_1998`.

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
        |NNLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(2)}`
    """
    S1 = c.get(c.S1, cache, n)
    S1m = S1 - 1 / n  # harmonic_S1(n - 1)

    D1 = -1.0 / n**2
    D11 = -1.0 / (n + 1.0) ** 2
    D2 = 2.0 / n**3
    D21 = 2.0 / (n + 1.0) ** 3

    a_gg_f = (
        -15.0
        - 8.0 / (n - 1.0)
        + 80.0 / n
        - 48.0 / (n + 1.0)
        - 24.0 / (n + 2.0)
        + 4.0 / 3.0 * (-6.0 / n**4 - 6.0 / (n + 1.0) ** 4)
        + 6.0 * D2
        + 10.0 * D21
        + 32.0 * D1
        + 48.0 * D11
    )
    a_gg_a = (
        -224.0 / 27.0 * (S1 - 1.0 / n)
        + 10.0 / 9.0
        + 4.0 / 3.0 * (S1 + 1.0 / (n + 1.0)) / (n + 1.0)
        + 1.0
        / 27.0
        * (556.0 / (n - 1.0) - 628.0 / n + 548.0 / (n + 1.0) - 700.0 / (n + 2.0))
        + 4.0 / 3.0 * (D2 + D21)
        + 1.0 / 9.0 * (52.0 * D1 + 88.0 * D11)
    )

    a_gg_l0 = constants.TR * (constants.CF * a_gg_f + constants.CA * a_gg_a)

    a_gg_l1 = (
        8
        / 3
        * (
            (
                8
                + 2 * n
                - 34 * n**2
                - 72 * n**3
                - 77 * n**4
                - 37 * n**5
                - 19 * n**6
                - 11 * n**7
                - 4 * n**8
            )
            / ((n * (n + 1)) ** 3 * (-2 + n + n**2))
            + 5 * S1m
        )
    )

    a_gg_l2 = (
        4
        / 9
        * (
            1
            + 6 * (2 + n + n**2) ** 2 / ((n * (n + 1)) ** 2 * (-2 + n + n**2))
            - 9 * (-4 - 3 * n + n**3) / (n * (n + 1) * (-2 + n + n**2))
        )
        - 4 * S1m
    )

    return a_gg_l2 * L**2 + a_gg_l1 * L + a_gg_l0


@nb.njit(cache=True)
def A_singlet(n, cache, L, is_msbar=False):
    r"""Compute the |NNLO| singlet |OME|.

    .. math::
        A^{S,(2)} = \left(\begin{array}{cc}
        A_{gg, H}^{S,(2)} & A_{gq, H}^{S,(2)} & 0 \\
        0 & A_{qq,H}^{NS,(2)} & 0\\
        A_{hg}^{S,(2)} & A_{hq}^{PS,(2)} & 0\\
        \end{array}\right)

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`
    is_msbar: bool
        add the |MSbar| contribution

    Returns
    -------
    numpy.ndarray
        |NNLO| singlet |OME| :math:`A^{S,(2)}(N)`
    """
    A_hq_2 = A_hq_ps(n, cache, L)
    A_qq_2 = A_qq_ns(n, cache, L)
    A_hg_2 = A_hg(n, cache, L)
    A_gq_2 = A_gq(n, cache, L)
    A_gg_2 = A_gg(n, cache, L)
    if is_msbar:
        A_hg_2 -= 2.0 * 4.0 * constants.CF * A_hg_1(n, L=1.0)
        A_gg_2 -= 2.0 * 4.0 * constants.CF * A_gg_1(L=1.0)
    A_S = np.array(
        [[A_gg_2, A_gq_2, 0.0], [0.0, A_qq_2, 0.0], [A_hg_2, A_hq_2, 0.0]],
        np.complex128,
    )
    return A_S


@nb.njit(cache=True)
def A_ns(n, cache, L):
    r"""Compute the |NNLO| non-singlet |OME|.

    .. math::
        A^{NS,(2)} = \left(\begin{array}{cc}
        A_{qq,H}^{NS,(2)} & 0 \\
        0 & 0 \\
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
        |NNLO| non-singlet |OME| :math:`A^{NS,(2)}`
    """
    return np.array([[A_qq_ns(n, cache, L), 0.0], [0 + 0j, 0 + 0j]], np.complex128)
