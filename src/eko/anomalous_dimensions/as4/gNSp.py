# -*- coding: utf-8 -*-
"""
This module contains the anomalous dimension :math:`\\gamma_{ns,+}^{(3)}`
"""
import numba as nb

from ...constants import CA, CF
from ...harmonics import w5 as sf
from ...harmonics.constants import zeta3, zeta4


@nb.njit(cache=True)
def gamma_ns_nf3(n, sx):
    """
    Implements the common part proportional to :math:`nf^3`,
    of :math:`\\gamma_{ns,+}^{(3)},`\\gamma_{ns,-}^{(3)},`\\gamma_{ns,v}^{(3)}`

    The expression is copied exact from eq. 3.6. of :cite:`Davies:2016jie`

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : np.ndarray
            List of harmonic sums: :math:`S_{1},S_{2},S_{3},S_{4}`

    Returns
    -------
        g_ns_nf3 : complex
            |N3LO| non-singlet anomalous dimension :math:`\\gamma_{ns}^{(3)}|_{nf^3}`
    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    eta = 1 / n * 1 / (n + 1)
    g_ns_nf3 = CF * (
        -32 / 27 * zeta3 * eta - 16 / 9 * zeta3 - 16 / 27 * eta
        ^ 4 - 16 / 81 * eta
        ^ 3 + 80 / 27 * eta
        ^ 2
        - 320 / 81 * eta
        + 32 / 27 * 1 / (n + 1) ** 4
        + 128 / 27 * 1 / (n + 1) ** 2
        + 64 / 27 * S1 * zeta3
        - 32 / 81 * S1
        - 32 / 81 * S2
        - 160 / 81 * S3
        + 32 / 27 * S4
        + 131 / 81
    )
    return g_ns_nf3


def A_3(n, sx):
    """
    Implements the common part proportional to :math:`nf^2`,
    of :math:`\\gamma_{ns,+}^{(3)},`\\gamma_{ns,-}^{(3)},`\\gamma_{ns,v}^{(3)}`

    The expression is taken from eq. 3.1 of :cite:`Davies:2016jie`

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache

    Returns
    -------
        A_3 : complex
            |N3LO| non-singlet anomalous dimension common part proportional to :math:`nf^2 C_F^2`
    """
    S1, _ = sx[0]
    S2, _ = sx[1]
    S3, S21, _, _, _, _ = sx[2]
    S4, S31, _, _, _, _, _ = sx[3]
    S5, _ = sx[4]
    S41 = sf.S41(n, S1, S2, S3)
    S311 = sf.S311(n, S1, S2)
    S23 = sf.S23(n, S1, S2, S3)
    return (
        127 / 9
        - 64 / (3 * (1 + n) ** 5)
        - 32 / (3 * n**5 * (1 + n) ** 5)
        - 304 / (9 * (1 + n) ** 4)
        - 7088 / (81 * (1 + n) ** 3)
        + 944 / (81 * n**3 * (1 + n) ** 3)
        - 13640 / (27 * (1 + n) ** 2)
        - 4550 / (81 * n**2 * (1 + n) ** 2)
        + 20681 / (81 * n * (1 + n))
        + (4238 * S1) / 81
        - (128 * S1) / (3 * (1 + n) ** 4)
        + (160 * S1) / (9 * n**4 * (1 + n) ** 4)
        - (128 * S1) / (9 * (1 + n) ** 3)
        + (32 * S1) / (3 * n**3 * (1 + n) ** 3)
        - (6784 * S1) / (81 * (1 + n) ** 2)
        - (7312 * S1) / (81 * n**2 * (1 + n) ** 2)
        + (6784 * S1) / (81 * n * (1 + n))
        - (10072 * S2) / 81
        + (128 * S2) / (9 * n**3 * (1 + n) ** 3)
        + (2048 * S2) / (27 * (1 + n) ** 2)
        + (32 * S2) / (27 * n**2 * (1 + n) ** 2)
        - (3328 * S2) / (81 * n * (1 + n))
        - (1216 * S21) / 81
        - (128 * S23) / 9
        + (9184 * S3) / 81
        - (128 * S3) / (9 * (1 + n) ** 2)
        + (64 * S3) / (9 * n**2 * (1 + n) ** 2)
        - (128 * S3) / (27 * n * (1 + n))
        - 1216 / 81 * (S1 * S2 - S21 + S3)
        + (640 * S31) / 9
        + (64 * S31) / (9 * n * (1 + n))
        - (848 * S4) / 9
        - (32 * S4) / (9 * n * (1 + n))
        + 640 / 27 * (S2**2 + S4)
        + 640 / 27 * (S1 * S3 - S31 + S4)
        - (320 * S41) / 9
        + (128 * S5) / 3
        - 256 / 9 * (-S23 + S2 * S3 + S5)
        + 64 / 9 * (S1 * S4 - S41 + S5)
        - 128 / 9 * (-S23 + S2 * S3 + S1 * S31 - 2 * S311 + S41 + S5)
        - (88 * zeta3) / 3
        - (64 * zeta3) / (3 * (1 + n) ** 2)
        + (32 * zeta3) / (3 * n**2 * (1 + n) ** 2)
        - (320 * zeta3) / (9 * n * (1 + n))
        + (640 * S1 * zeta3) / 9
        - (64 * S2 * zeta3) / 3
        + 24 * zeta4
        + (16 * zeta4) / (n * (1 + n))
        - 32 * S1 * zeta4
    )


def B_3p(n, sx):
    """
    Parametrization of eq. 3.2 of :cite:`Davies:2016jie`.
    This contribution is subleading with respect to A_3.
    The exact expression contains weight 5 harmonics sum not yet
    implemented in eko.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache

    Returns
    -------
        B_3p : complex
            |N3LO| singlet-like non-singlet anomalous dimension part
            proportional to :math:`C_F (C_A - 2 C_F) nf^2`
    """
    S1 = sx[0][0]
    return -52.6196 + 58.6005 * S1 - 28.9527 * S1 / n ^ 2 + 16.6447 * S1 / n


@nb.njit(cache=True)
def gamma_nsp_nf2(n, sx):
    """
    Implements the singlet-like non-singlet part proportional to :math:`nf^2`
    as in eq. 2.12 of :cite:`Davies:2016jie`.
    Note: A_3 is already multiplied by a factor of 2.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache

    Returns
    -------
        g_ns_nf2 : complex
            |N3LO| singlet-like non-singlet anomalous dimension :math:`\\gamma_{ns,+}^{(3)}|_{nf^2}`

    See Also
    --------
        A_3: :math:`A^{(3)}`
        B_3p: :math:`B^{(3)}_{+}`
    """
    return CF**2 * A_3(n, sx) + CF * (CA - 2 * CF) * B_3p(n, sx)


@nb.njit(cache=True)
def gamma_nsp_nf1(n, sx):
    return 0


@nb.njit(cache=True)
def gamma_nsp_nf0(n, sx):
    return 0


@nb.njit(cache=True)
def gamma_nsp(n, nf, sx):
    """
    Computes the |N3LO| singlet-like non-singlet anomalous dimension.

    Parameters
    ----------
        n : complex
            Mellin moment
        nf : int
            Number of active flavors
        sx : list
            harmonic sums cache

    Returns
    -------
        gamma_nsp : complex
            |N3LO| singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,+}^{(3)}(N)`

    See Also
    --------
        gamma_nsp_nf0: :math:`\\gamma_{ns,+}^{(3)}|_{nf^0}`
        gamma_nsp_nf1: :math:`\\gamma_{ns,+}^{(3)}|_{nf^1}`
        gamma_nsp_nf2: :math:`\\gamma_{ns,+}^{(3)}|_{nf^2}`
        gamma_ns_nf3: :math:`\\gamma_{ns}^{(3)}|_{nf^3}`
    """
    return (
        gamma_nsp_nf0(n, sx)
        + nf * gamma_nsp_nf1(n, sx)
        + nf**2 * gamma_nsp_nf2(n, sx)
        + nf**3 * gamma_ns_nf3(n, sx)
    )
