r"""This module contains the anomalous dimension :math:`\gamma_{gg}^{(3)}`
"""
import numba as nb
import numpy as np

from .....harmonics.log_functions import lm11


@nb.njit(cache=True)
def gamma_gg_nf3(n, sx):
    r"""Implements the part proportional to :math:`nf^3` of :math:`\gamma_{gg}^{(3)}`,
    the expression is copied exact from Eq. 3.14 of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^3}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3, S21, _, _, _, _ = sx[2]
    return 3.0 * (
        -0.0205761316872428
        + 2.599239604033225 / (-1.0 + n)
        - 1.1851851851851851 / np.power(n, 4)
        - 3.753086419753086 / np.power(n, 3)
        - 5.679012345679013 / np.power(n, 2)
        - 2.8050009209056537 / n
        - 1.1851851851851851 / np.power(1.0 + n, 4)
        + 3.753086419753086 / np.power(1.0 + n, 3)
        - 5.679012345679013 / np.power(1.0 + n, 2)
        + 2.8050009209056537 / (1.0 + n)
        - 2.599239604033225 / (2.0 + n)
        + 2.454258338353606 * S1
        - (8.674897119341564 * S1) / (-1.0 + n)
        + (2.3703703703703702 * S1) / np.power(n, 3)
        + (7.506172839506172 * S1) / np.power(n, 2)
        + (7.901234567901234 * S1) / n
        - (2.3703703703703702 * S1) / np.power(1.0 + n, 3)
        + (7.506172839506172 * S1) / np.power(1.0 + n, 2)
        - (7.901234567901234 * S1) / (1.0 + n)
        + (8.674897119341564 * S1) / (2.0 + n)
        - (2.567901234567901 * S2) / (-1.0 + n)
        + (2.3703703703703702 * S2) / np.power(n, 2)
        + (1.6296296296296295 * S2) / n
        + (2.3703703703703702 * S2) / np.power(1.0 + n, 2)
        - (1.6296296296296295 * S2) / (1.0 + n)
        + (2.567901234567901 * S2) / (2.0 + n)
        + (2.567901234567901 * (np.power(S1, 2) + S2)) / (-1.0 + n)
        - (2.3703703703703702 * (np.power(S1, 2) + S2)) / np.power(n, 2)
        - (1.6296296296296295 * (np.power(S1, 2) + S2)) / n
        - (2.3703703703703702 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 2)
        + (1.6296296296296295 * (np.power(S1, 2) + S2)) / (1.0 + n)
        - (2.567901234567901 * (np.power(S1, 2) + S2)) / (2.0 + n)
    ) + 1.3333333333333333 * (
        -0.6337448559670782
        - 54.23172286962781 / (-1.0 + n)
        + 3.5555555555555554 / np.power(n, 5)
        + 13.62962962962963 / np.power(n, 4)
        + 27.65432098765432 / np.power(n, 3)
        + 61.00190529209603 / np.power(n, 2)
        + 26.28917081074211 / n
        + 10.666666666666666 / np.power(1.0 + n, 5)
        - 31.40740740740741 / np.power(1.0 + n, 4)
        + 31.604938271604937 / np.power(1.0 + n, 3)
        - 12.479576189385448 / np.power(1.0 + n, 2)
        + 44.82194030036901 / (1.0 + n)
        - 16.879388241483305 / (2.0 + n)
        + (48.724279835390945 * S1) / (-1.0 + n)
        - (7.111111111111111 * S1) / np.power(n, 4)
        - (27.25925925925926 * S1) / np.power(n, 3)
        - (36.34567901234568 * S1) / np.power(n, 2)
        - (56.49382716049383 * S1) / n
        + (7.111111111111111 * S1) / np.power(1.0 + n, 4)
        - (17.77777777777778 * S1) / np.power(1.0 + n, 3)
        + (3.950617283950617 * S1) / np.power(1.0 + n, 2)
        + (4.345679012345679 * S1) / (1.0 + n)
        + (3.4238683127572016 * S1) / (2.0 + n)
        + (27.65432098765432 * S2) / (-1.0 + n)
        - (7.111111111111111 * S2) / np.power(n, 3)
        - (27.25925925925926 * S2) / np.power(n, 2)
        - (16.59259259259259 * S2) / n
        + (7.111111111111111 * S2) / np.power(1.0 + n, 3)
        - (27.25925925925926 * S2) / np.power(1.0 + n, 2)
        + (16.59259259259259 * S2) / (1.0 + n)
        - (27.65432098765432 * S2) / (2.0 + n)
        - (15.012345679012345 * (np.power(S1, 2) + S2)) / (-1.0 + n)
        + (7.111111111111111 * (np.power(S1, 2) + S2)) / np.power(n, 3)
        + (8.296296296296296 * (np.power(S1, 2) + S2)) / np.power(n, 2)
        + (22.51851851851852 * (np.power(S1, 2) + S2)) / n
        + (4.7407407407407405 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 2)
        - (15.407407407407407 * (np.power(S1, 2) + S2)) / (1.0 + n)
        + (7.901234567901234 * (np.power(S1, 2) + S2)) / (2.0 + n)
        - (9.481481481481481 * S21) / (-1.0 + n)
        + (14.222222222222221 * S21) / np.power(n, 2)
        + (21.333333333333332 * S21) / n
        + (14.222222222222221 * S21) / np.power(1.0 + n, 2)
        - (21.333333333333332 * S21) / (1.0 + n)
        - (28.444444444444443 * S21) / (n * (1.0 + n))
        + (9.481481481481481 * S21) / (2.0 + n)
        + (4.7407407407407405 * S3) / (-1.0 + n)
        - (7.111111111111111 * S3) / np.power(n, 2)
        - (10.666666666666666 * S3) / n
        - (7.111111111111111 * S3) / np.power(1.0 + n, 2)
        + (10.666666666666666 * S3) / (1.0 + n)
        + (14.222222222222221 * S3) / (n * (1.0 + n))
        - (4.7407407407407405 * S3) / (2.0 + n)
        - (9.481481481481481 * (S1 * S2 - 1.0 * S21 + S3)) / (-1.0 + n)
        + (14.222222222222221 * (S1 * S2 - 1.0 * S21 + S3)) / np.power(n, 2)
        + (21.333333333333332 * (S1 * S2 - 1.0 * S21 + S3)) / n
        + (14.222222222222221 * (S1 * S2 - 1.0 * S21 + S3)) / np.power(1.0 + n, 2)
        - (21.333333333333332 * (S1 * S2 - 1.0 * S21 + S3)) / (1.0 + n)
        - (28.444444444444443 * (S1 * S2 - 1.0 * S21 + S3)) / (n * (1.0 + n))
        + (9.481481481481481 * (S1 * S2 - 1.0 * S21 + S3)) / (2.0 + n)
        + (1.5802469135802468 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (-1.0 + n)
        - (2.3703703703703702 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / np.power(n, 2)
        - (3.5555555555555554 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3)) / n
        - (2.3703703703703702 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / np.power(1.0 + n, 2)
        + (3.5555555555555554 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (1.0 + n)
        + (4.7407407407407405 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (n * (1.0 + n))
        - (1.5802469135802468 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (2.0 + n)
    )


@nb.njit(cache=True)
def gamma_gg_nf1(n, sx):
    r"""Implements the part proportional to :math:`nf^1` of :math:`\gamma_{gg}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    return (
        18371.82290926215
        + 1992.766087237516 / np.power(-1.0 + n, 3)
        - 10452.576219113906 / np.power(-1.0 + n, 2)
        + 19915.749052258507 / (-1.0 + n)
        + 20005.925925925927 / np.power(n, 7)
        - 19449.679012345678 / np.power(n, 6)
        + 80274.123066115 / np.power(n, 5)
        - 11714.245609287387 * S1
        + 18617.18939338288 * lm11(n, S1)
    )


@nb.njit(cache=True)
def gamma_gg_nf2(n, sx):
    r"""Implements the part proportional to :math:`nf^2` of :math:`\gamma_{gg}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^2}`

    """
    S1 = sx[0][0]
    return (
        -436.18166675733164
        + 18.346203400819753 / np.power(-1.0 + n, 2)
        - 728.2679478666736 / (-1.0 + n)
        - 568.8888888888889 / np.power(n, 7)
        + 1725.6296296296296 / np.power(n, 6)
        - 2196.543209876543 / np.power(n, 5)
        + 440.0487580115612 * S1
        - 382.0574816096663 * lm11(n, S1)
    )


@nb.njit(cache=True)
def gamma_gg_nf0(n, sx):
    r"""Implements the part proportional to :math:`nf^0` of :math:`\gamma_{gg}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^0}`

    """
    S1 = sx[0][0]
    return (
        -61782.868048046716
        - 49851.703887834694 / np.power(-1.0 + n, 4)
        + 213823.9810748423 / np.power(-1.0 + n, 3)
        - 241732.26024803068 / np.power(-1.0 + n, 2)
        + 88694.20340182834 / (-1.0 + n)
        - 103680.0 / np.power(n, 7)
        - 17280.0 / np.power(n, 6)
        - 627978.8224813186 / np.power(n, 5)
        + 40880.33011934297 * S1
        - 13643.320974357539 * lm11(n, S1)
    )


@nb.njit(cache=True)
def gamma_gg(n, nf, sx):
    r"""Computes the |N3LO| gluon-gluon singlet anomalous dimension.

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
    complex
        |N3LO| gluon-gluon singlet anomalous dimension
        :math:`\gamma_{gg}^{(3)}(N)`

    See Also
    --------
    gamma_gg_nf0: :math:`\gamma_{gg}^{(3)}|_{nf^0}`
    gamma_gg_nf1: :math:`\gamma_{gg}^{(3)}|_{nf^1}`
    gamma_gg_nf2: :math:`\gamma_{gg}^{(3)}|_{nf^2}`
    gamma_gg_nf3: :math:`\gamma_{gg}^{(3)}|_{nf^3}`

    """
    return (
        gamma_gg_nf0(n, sx)
        + nf * gamma_gg_nf1(n, sx)
        + nf**2 * gamma_gg_nf2(n, sx)
        + nf**3 * gamma_gg_nf3(n, sx)
    )
