# -*- coding: utf-8 -*-
r"""
This module contains the |NNLO| |OME| for the matching conditions in the |VFNS|
(see, :cite:`Buza_1998` appendix B).
The expession for :math:`\mu_F^2 = m_H^2` are taken from :cite:`Vogt:2004ns` directly in N space.
While the parts proportional to :math:`\ln(\mu_F^2 / m_h^2)` comes |QCDNUM|
(https://github.com/N3PDF/external/blob/master/qcdnum/qcdnum/pij/ome.f)
and Mellin transformed with Mathematica.
"""
import numba as nb
import numpy as np

from .. import constants
from ..anomalous_dimensions.harmonics import zeta2, zeta3, harmonic_S2


@nb.njit("c16(c16,c16[:],f8)", cache=True)
def A_qq_2_ns(n, sx, L):
    r"""
    |NNLO| light-light non-singlet |OME| :math:`A_{qq,H}^{NS,(2)}` given in
    Eq. (B.4) of :cite:`Buza_1998`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : numpy.ndarray
            List of harmonic sums
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_qq_2_ns : complex
            |NNLO| light-light non-singlet |OME| :math:`A_{qq,H}^{NS,(2)}`
    """
    S1 = sx[0]
    S2 = sx[1]
    S3 = sx[2]
    S1m = S1 - 1 / n  # harmonic_S1(n - 1)
    S2m = S2 - 1 / n ** 2  # harmonic_S2(n - 1)

    a_qq_2_l0 = (
        -224.0 / 27.0 * (S1 - 1.0 / n)
        - 8.0 / 3.0 * zeta3
        + 40 / 9.0 * zeta2
        + 73.0 / 18.0
        + 44.0 / 27.0 / n
        - 268.0 / 27.0 / (n + 1.0)
        + 8.0 / 3.0 * (-1.0 / n ** 2 + 1.0 / (n + 1.0) ** 2)
        + 20.0 / 9.0 * (S2 - 1.0 / n ** 2 - zeta2 + S2 + 1.0 / (n + 1.0) ** 2 - zeta2)
        + 2.0
        / 3.0
        * (
            -2.0 * (S3 - 1.0 / n ** 3 - zeta3)
            - 2.0 * (S3 + 1.0 / (n + 1.0) ** 3 - zeta3)
        )
    )

    a_qq_2_l1 = (
        2
        * (-12 - 28 * n + 9 * n ** 2 + 34 * n ** 3 - 3 * n ** 4)
        / (9 * (n * (n + 1)) ** 2)
        + 80 / 9 * S1m
        - 16 / 3 * S2m
    )

    a_qq_2_l2 = -2 * ((2 + n - 3 * n ** 2) / (3 * n * (n + 1)) + 4 / 3 * S1m)
    return (
        constants.CF * constants.TR * (a_qq_2_l2 * L ** 2 + a_qq_2_l1 * L + a_qq_2_l0)
    )


@nb.njit("c16(c16,c16[:],f8)", cache=True)
def A_hq_2_ps(n, sx, L):
    r"""
    |NNLO| heavy-light pure-singlet |OME| :math:`A_{Hq}^{PS,(2)}` given in
    Eq. (B.1) of :cite:`Buza_1998`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : numpy.ndarray
            List of harmonic sums
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_hq_2_ps : complex
            |NNLO| heavy-light pure-singlet |OME| :math:`A_{Hq}^{PS,(2)}`
    """
    S2 = sx[1]

    F1M = 1.0 / (n - 1.0) * (zeta2 - (S2 - 1.0 / n ** 2))
    F11 = 1.0 / (n + 1.0) * (zeta2 - (S2 + 1.0 / (n + 1.0) ** 2))
    F12 = 1.0 / (n + 2.0) * (zeta2 - (S2 + 1.0 / (n + 1.0) ** 2 + 1.0 / (n + 2.0) ** 2))
    F21 = -F11 / (n + 1.0)
    a_hq_2_l0 = (
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
        - 4.0 / 3.0 * (-6.0 / n ** 4 - 6.0 / (n + 1.0) ** 4)
        + 2.0 * 2.0 / n ** 3
        + 10.0 * 2.0 / (n + 1.0) ** 3
        + 16.0 / 3.0 * 2.0 / (n + 2.0) ** 3
        - 16.0 * zeta2 * (-1.0 / n ** 2 - 1.0 / (n + 1.0) ** 2)
        + 56.0 / 3.0 / n ** 2
        + 88.0 / 3.0 / (n + 1.0) ** 2
        + 448.0 / 9.0 / (n + 2.0) ** 2
        + 32.0 / 3.0 * F1M
        + 8.0 * ((zeta2 - S2) / n - F11)
        - 32.0 / 3.0 * F12
        + 16.0 * (-(zeta2 - S2) / n ** 2 + F21)
    )

    a_hq_2_l1 = (
        8
        * (2 + n * (5 + n))
        * (4 + n * (4 + n * (7 + 5 * n)))
        / ((n - 1) * (n + 2) ** 2 * (n * (n + 1)) ** 3)
    )

    a_hq_2_l2 = -4 * (2 + n + n ** 2) ** 2 / ((n - 1) * (n + 2) * (n * (n + 1)) ** 2)

    return (
        constants.CF * constants.TR * (a_hq_2_l2 * L ** 2 + a_hq_2_l1 * L + a_hq_2_l0)
    )


@nb.njit("c16(c16,c16[:],f8)", cache=True)
def A_hg_2(n, sx, L):
    r"""
    |NNLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(2)}` given in
    Eq. (B.3) of :cite:`Buza_1998`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : numpy.ndarray
            List of harmonic sums
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_hg_2 : complex
            |NNLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(2)}`
    """
    S1 = sx[0]
    S2 = sx[1]
    S3 = sx[2]
    S1m = S1 - 1 / n  # harmonic_S1(n - 1)
    S2m = S2 - 1 / n ** 2  # harmonic_S2(n - 1)
    Sp2m = harmonic_S2((n - 1) / 2)
    Sp2p = harmonic_S2(n / 2)

    E2 = 2.0 / n * (zeta3 - S3 + 1.0 / n * (zeta2 - S2 - S1 / n))

    a_hg_2_l0 = (
        -0.006
        + 1.111 * (S1 ** 3 + 3.0 * S1 * S2 + 2.0 * S3) / n
        - 0.400 * (S1 ** 2 + S2) / n
        + 2.770 * S1 / n
        - 24.89 / (n - 1.0)
        - 187.8 / n
        + 249.6 / (n + 1.0)
        + 1.556 * 6.0 / n ** 4
        - 3.292 * 2.0 / n ** 3
        + 93.68 * 1.0 / n ** 2
        - 146.8 * E2
    )

    a_hg_2_l1 = (
        2
        * (
            +640
            + 2192 * n
            + 2072 * n ** 2
            + 868 * n ** 3
            + 518 * n ** 4
            + 736 * n ** 5
            + 806 * n ** 6
            + 542 * n ** 7
            + 228 * n ** 8
            + 38 * n ** 9
        )
        / (3 * (n * (n + 1) * (n + 2)) ** 3 * (n - 1))
    )

    a_hg_2_l1 -= (
        2
        * (
            n
            * (n ** 2 - 1)
            * (n + 2)
            * (
                4 * (36 + n * (88 + n * (33 + n * (8 + 9 * n)))) * S1m
                + n
                * (n + 1)
                * (n + 2)
                * (2 + n + n ** 2)
                * (10 * S1m ** 2 - 9 * Sp2m + 26 * S2m + 9 * Sp2p)
            )
        )
        / (3 * (n * (n + 1) * (n + 2)) ** 3 * (n - 1))
    )
    a_hg_2_l1 += 12 * zeta2 * (-2 + n + n ** 3) / (n * (n ** 2 - 1) * (n + 2))

    a_hg_2_l2 = (
        4
        * (2 + n + n ** 2)
        * (2 * (-11 + n + n ** 2) * (1 + n + n ** 2) / (n - 1))
        / (3 * (n * (n + 1) * (n + 2)) ** 2)
    ) + 20 * (2 + n + n ** 2) * S1 / (3 * n * (n + 1) * (n + 2))

    return a_hg_2_l2 * L ** 2 + a_hg_2_l1 * L + a_hg_2_l0


@nb.njit("c16(c16,c16[:],f8)", cache=True)
def A_gq_2(n, sx, L):
    r"""
    |NNLO| gluon-quark |OME| :math:`A_{gq,H}^{S,(2)}` given in
    Eq. (B.5) of :cite:`Buza_1998`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : numpy.ndarray
            List of harmonic sums
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_gq_2 : complex
            |NNLO| gluon-quark |OME| :math:`A_{gq,H}^{S,(2)}`
    """
    S1 = sx[0]
    S2 = sx[1]
    S1m = S1 - 1 / n  # harmonic_S1(n - 1)

    B2M = ((S1 - 1.0 / n) ** 2 + S2 - 1.0 / n ** 2) / (n - 1.0)
    B21 = ((S1 + 1.0 / (n + 1.0)) ** 2 + S2 + 1.0 / (n + 1.0) ** 2) / (n + 1.0)

    a_gq_2_l0 = (
        4.0 / 3.0 * (2.0 * B2M - 2.0 * (S1 ** 2 + S2) / n + B21)
        + 8.0
        / 9.0
        * (
            -10.0 * (S1 - 1.0 / n) / (n - 1.0)
            + 10.0 * S1 / n
            - 8.0 * (S1 + 1.0 / (n + 1.0)) / (n + 1.0)
        )
        + 1.0 / 27.0 * (448.0 * (1.0 / (n - 1.0) - 1.0 / n) + 344.0 / (n + 1.0))
    )

    a_gq_2_l1 = -(
        -96
        + 16 * n * (7 + n * (21 + 10 * n + 8 * n ** 2))
        - 48 * n * (1 + n) * (2 + n + n ** 2) * S1m
    ) / (9 * (n - 1) * (n * (1 + n)) ** 2)

    a_gq_2_l2 = 8 * (2 + n + n ** 2) / (3 * n * (n ** 2 - 1))

    return (
        constants.CF * constants.TR * (a_gq_2_l2 * L ** 2 + a_gq_2_l1 * L + a_gq_2_l0)
    )


@nb.njit("c16(c16,c16[:],f8)", cache=True)
def A_gg_2(n, sx, L):
    r"""
    |NNLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(2)} ` given in
    Eq. (B.7) of :cite:`Buza_1998`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : numpy.ndarray
            List of harmonic sums
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_gg_2 : complex
            |NNLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(2)}`
    """
    S1 = sx[0]
    S1m = S1 - 1 / n  # harmonic_S1(n - 1)

    D1 = -1.0 / n ** 2
    D11 = -1.0 / (n + 1.0) ** 2
    D2 = 2.0 / n ** 3
    D21 = 2.0 / (n + 1.0) ** 3

    a_gg_2f = (
        -15.0
        - 8.0 / (n - 1.0)
        + 80.0 / n
        - 48.0 / (n + 1.0)
        - 24.0 / (n + 2.0)
        + 4.0 / 3.0 * (-6.0 / n ** 4 - 6.0 / (n + 1.0) ** 4)
        + 6.0 * D2
        + 10.0 * D21
        + 32.0 * D1
        + 48.0 * D11
    )
    a_gg_2a = (
        -224.0 / 27.0 * (S1 - 1.0 / n)
        + 10.0 / 9.0
        + 4.0 / 3.0 * (S1 + 1.0 / (n + 1.0)) / (n + 1.0)
        + 1.0
        / 27.0
        * (556.0 / (n - 1.0) - 628.0 / n + 548.0 / (n + 1.0) - 700.0 / (n + 2.0))
        + 4.0 / 3.0 * (D2 + D21)
        + 1.0 / 9.0 * (52.0 * D1 + 88.0 * D11)
    )

    a_gg_2_l0 = constants.TR * (constants.CF * a_gg_2f + constants.CA * a_gg_2a)

    a_gg_2_l1 = (
        8
        / 3
        * (
            (
                8
                + 2 * n
                - 34 * n ** 2
                - 72 * n ** 3
                - 77 * n ** 4
                - 37 * n ** 5
                - 19 * n ** 6
                - 11 * n ** 7
                - 4 * n ** 8
            )
            / ((n * (n + 1)) ** 3 * (-2 + n + n ** 2))
            + 5 * S1m
        )
    )

    a_gg_2_l2 = (
        4
        / 9
        * (
            1
            + 6 * (2 + n + n ** 2) ** 2 / ((n * (n + 1)) ** 2 * (-2 + n + n ** 2))
            - 9 * (-4 - 3 * n + n ** 3) / (n * (n + 1) * (-2 + n + n ** 2))
        )
        - 4 * S1m
    )

    return a_gg_2_l2 * L ** 2 + a_gg_2_l1 * L + a_gg_2_l0


@nb.njit("c16[:,:](c16,c16[:],f8)", cache=True)
def A_singlet_2(n, sx, L):
    r"""
      Computes the |NNLO| singlet |OME|.

      .. math::
          A^{S,(2)} = \left(\begin{array}{cc}
            A_{gg, H}^{S,(2)} & A_{gq, H}^{S,(2)} & 0
            0 & A_{qq,H}^{NS,(2)} & 0\\
            A_{hg}^{S,(2)} & A_{hq}^{PS,(2)} & 0\\
          \end{array}\right)

      Parameters
      ----------
        n : complex
            Mellin moment
        sx : numpy.ndarray
            List of harmonic sums
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

      Returns
      -------
        A_S_2 : numpy.ndarray
            |NNLO| singlet |OME| :math:`A^{S,(2)}(N)`

      See Also
      --------
        A_ns_2 : :math:`A_{qq,H}^{NS,(2)}`
        A_hq_2 : :math:`A_{hq}^{PS,(2)}`
        A_hg_2 : :math:`A_{hg}^{S,(2)}`
        A_gq_2 : :math:`A_{gq, H}^{S,(2)}`
        A_gg_2 : :math:`A_{gg, H}^{S,(2)}`
    """
    A_hq = A_hq_2_ps(n, sx, L)
    A_qq = A_qq_2_ns(n, sx, L)
    A_hg = A_hg_2(n, sx, L)
    A_gq = A_gq_2(n, sx, L)
    A_gg = A_gg_2(n, sx, L)
    A_S_2 = np.array(
        [[A_gg, A_gq, 0.0], [0.0, A_qq, 0.0], [A_hg, A_hq, 0.0]], np.complex_
    )
    return A_S_2


@nb.njit("c16[:,:](c16,c16[:],f8)", cache=True)
def A_ns_2(n, sx, L):
    r"""
      Computes the |NNLO| non-singlet |OME|.

      .. math::
          A^{NS,(2)} = \left(\begin{array}{cc}
            A_{qq,H}^{NS,(2)} & 0\\
            0 & 0\\
          \end{array}\right)

      Parameters
      ----------
        n : complex
            Mellin moment
        sx : numpy.ndarray
            List of harmonic sums
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

      Returns
      -------
        A_NS_2 : numpy.ndarray
            |NNLO| non-singlet |OME| :math:`A^{NS,(2)}`

      See Also
      --------
        A_qq_2_ns : :math:`A_{qq,H}^{NS,(2)}`
    """
    A_qq = A_qq_2_ns(n, sx, L)
    return np.array([[A_qq, 0.0], [0+0j, 0+0j]], np.complex_)
