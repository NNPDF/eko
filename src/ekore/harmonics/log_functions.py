r"""Implementation of Mellin transformation of logarithms.

We provide transforms of:

- :math:`(1-x)\ln^k(1-x), \quad k = 1,2,3`
- :math:`\ln^k(1-x), \quad k = 1,3,4,5`

"""
import numba as nb

from .constants import zeta3


@nb.njit(cache=True)
def lm11m1(n, S1):
    r"""Mellin transform of :math:`(1-x)\ln(1-x)`.

    Parameters
    ----------
    n : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`

    Returns
    -------
    complex
        :math:`\mathcal{M}[(1-x)\ln(1-x)](N)`
    """
    return 1 / (1 + n) ** 2 - S1 / (1 + n) ** 2 - S1 / (n * (1 + n) ** 2)


@nb.njit(cache=True)
def lm12m1(n, S1, S2):
    r"""Mellin transform of :math:`(1-x)\ln^2(1-x)`.

    Parameters
    ----------
    n : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`
    S2 : complex
        Harmonic sum :math:`S_{2}(N)`

    Returns
    -------
    complex
        :math:`\mathcal{M}[(1-x)\ln^2(1-x)](N)`

    """
    return (
        -2 / (1 + n) ** 3
        - (2 * S1) / (1 + n) ** 2
        + S1**2 / n
        - S1**2 / (1 + n)
        + S2 / n
        - S2 / (1 + n)
    )


@nb.njit(cache=True)
def lm13m1(n, S1, S2, S3):
    r"""Mellin transform of :math:`(1-x)\ln^3(1-x)`.

    Parameters
    ----------
    n : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`
    S2 : complex
        Harmonic sum :math:`S_{2}(N)`
    S3 : complex
        Harmonic sum :math:`S_{3}(N)`

    Returns
    -------
    complex
        :math:`\mathcal{M}[(1-x)\ln^3(1-x)](N)`

    """
    return (
        6 / (1 + n) ** 4
        + (6 * S1) / (1 + n) ** 3
        + (3 * S1**2) / (1 + n) ** 2
        - S1**3 / n
        + S1**3 / (1 + n)
        + (3 * S2) / (1 + n) ** 2
        - (3 * S1 * S2) / n
        + (3 * S1 * S2) / (1 + n)
        - (2 * (6 * (2 * S3 - 2 * zeta3) + zeta3)) / n
        + (2 * (6 * (2 * S3 - 2 * zeta3) + zeta3)) / (1 + n)
    )


@nb.njit(cache=True)
def lm11(n, S1):
    r"""Mellin transform of :math:`\ln(1-x)`.

    Parameters
    ----------
    n : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`

    Returns
    -------
    complex
        :math:`\mathcal{M}[\ln(1-x)](N)`

    """
    return -S1 / n


@nb.njit(cache=True)
def lm13(n, S1, S2, S3):
    r"""Mellin transform of :math:`\ln^3(1-x)`.

    Parameters
    ----------
    n : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`
    S2 : complex
        Harmonic sum :math:`S_{2}(N)`
    S3 : complex
        Harmonic sum :math:`S_{3}(N)`

    Returns
    -------
    complex
        :math:`\mathcal{M}[\ln^3(1-x)](N)`

    """
    return -((S1**3 + 3 * S1 * S2 + 2 * S3) / n)


@nb.njit(cache=True)
def lm14(n, S1, S2, S3, S4):
    r"""Mellin transform of :math:`\ln^4(1-x)`.

    Parameters
    ----------
    n : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`
    S2 : complex
        Harmonic sum :math:`S_{2}(N)`
    S3 : complex
        Harmonic sum :math:`S_{3}(N)`
    S4 : complex
        Harmonic sum :math:`S_{4}(N)`

    Returns
    -------
    complex
        :math:`\mathcal{M}[\ln^4(1-x)](N)`

    """
    return (S1**4 + 6 * S1**2 * S2 + 3 * S2**2 + 8 * S1 * S3 + 6 * S4) / n


@nb.njit(cache=True)
def lm15(n, S1, S2, S3, S4, S5):
    r"""Mellin transform of :math:`\ln^5(1-x)`.

    Parameters
    ----------
    n : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`
    S2 : complex
        Harmonic sum :math:`S_{2}(N)`
    S3 : complex
        Harmonic sum :math:`S_{3}(N)`
    S4 : complex
        Harmonic sum :math:`S_{4}(N)`
    S5 : complex
        Harmonic sum :math:`S_{5}(N)`

    Returns
    -------
    complex
        :math:`\mathcal{M}[\ln^5(1-x)](N)`

    """
    return (
        -(
            S1**5
            + 10 * S1**3 * S2
            + 20 * S1**2 * S3
            + 15 * S1 * (S2**2 + 2 * S4)
            + 4 * (5 * S2 * S3 + 6 * S5)
        )
        / n
    )
