r"""Implementation of Mellin transformation of logarithms.

We provide transforms of:

- :math:`(1-x)\ln^k(1-x), \quad k = 1,2,3`
- :math:`\ln^k(1-x), \quad k = 1,3,4,5`
"""

import numba as nb


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
        3 * n * (1 + n) ** 2 * S1**2
        - (1 + n) ** 3 * S1**3
        + 3 * n * (1 + n) ** 2 * S2
        - 3 * (1 + n) * S1 * (-2 * n + (1 + n) ** 2 * S2)
        - 2 * (-3 * n + (1 + n) ** 3 * S3)
    ) / (n * (1 + n) ** 4)


@nb.njit(cache=True)
def lm14m1(n, S1, S2, S3, S4):
    r"""Mellin transform of :math:`(1-x)\ln^4(1-x)`.

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
        :math:`\mathcal{M}[(1-x)\ln^4(1-x)](N)`
    """
    return (
        -24 * n
        - 4 * n * (1 + n) ** 3 * S1**3
        + (1 + n) ** 4 * S1**4
        - 12 * n * (1 + n) ** 2 * S2
        + 3 * (1 + n) ** 4 * S2**2
        + 6 * (1 + n) ** 2 * S1**2 * (-2 * n + (1 + n) ** 2 * S2)
        - 8 * n * S3
        - 24 * n**2 * S3
        - 24 * n**3 * S3
        - 8 * n**4 * S3
        - 4
        * (1 + n)
        * S1
        * (3 * n * (1 + n) ** 2 * S2 - 2 * (-3 * n + (1 + n) ** 3 * S3))
        + 6 * S4
        + 24 * n * S4
        + 36 * n**2 * S4
        + 24 * n**3 * S4
        + 6 * n**4 * S4
    ) / (n * (1 + n) ** 5)


@nb.njit(cache=True)
def lm15m1(n, S1, S2, S3, S4, S5):
    r"""Mellin transform of :math:`(1-x)\ln^5(1-x)`.

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
        :math:`\mathcal{M}[(1-x)\ln^5(1-x)](N)`
    """
    return (1 / (n * (1 + n) ** 6)) * (
        5 * n * (1 + n) ** 4 * S1**4
        - (1 + n) ** 5 * S1**5
        + 15 * n * (1 + n) ** 4 * S2**2
        - 10 * (1 + n) ** 3 * S1**3 * (-2 * n + (1 + n) ** 2 * S2)
        + 40 * n * (1 + n) ** 3 * S3
        - 20 * (1 + n) ** 2 * S2 * (-3 * n + (1 + n) ** 3 * S3)
        + 10
        * S1**2
        * (3 * n * (1 + n) ** 4 * S2 - 2 * (1 + n) ** 2 * (-3 * n + (1 + n) ** 3 * S3))
        + 30 * n * (1 + n) ** 4 * S4
        - 5
        * (1 + n)
        * S1
        * (
            -12 * n * (1 + n) ** 2 * S2
            + 3 * (1 + n) ** 4 * S2**2
            - 8 * n * (1 + n) ** 3 * S3
            + 6 * (-4 * n + (1 + n) ** 4 * S4)
        )
        - 24 * (-5 * n + (1 + n) ** 5 * S5)
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
def lm12(n, S1, S2):
    r"""Mellin transform of :math:`\ln^2(1-x)`.

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
        :math:`\mathcal{M}[\ln^2(1-x)](N)`
    """
    return (S1**2 + S2) / n


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


@nb.njit(cache=True)
def lm11m2(n, S1):
    r"""Mellin transform of :math:`(1-x)^2\ln(1-x)`.

    Parameters
    ----------
    n : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`

    Returns
    -------
    complex
        :math:`\mathcal{M}[(1-x)^2\ln(1-x)](N)`
    """
    return (5 + 3 * n - (2 * (1 + n) * (2 + n) * S1) / n) / (
        (1 + n) ** 2 * (2 + n) ** 2
    )


@nb.njit(cache=True)
def lm12m2(n, S1, S2):
    r"""Mellin transform of :math:`(1-x)^2\ln^2(1-x)`.

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
        :math:`\mathcal{M}[(1-x)^2\ln^2(1-x)](N)`
    """
    return (
        2
        * (
            n * (-9 - 8 * n + n**3)
            - n * (10 + 21 * n + 14 * n**2 + 3 * n**3) * S1
            + (2 + 3 * n + n**2) ** 2 * S1**2
            + (2 + 3 * n + n**2) ** 2 * S2
        )
    ) / (n * (1 + n) ** 3 * (2 + n) ** 3)


@nb.njit(cache=True)
def lm13m2(n, S1, S2, S3):
    r"""Mellin transform of :math:`(1-x)^2\ln^3(1-x)`.

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
        :math:`\mathcal{M}[(1-x)^2\ln^3(1-x)](N)`
    """
    return (
        -6 * n * (-17 - 21 * n - 2 * n**2 + 6 * n**3 + 2 * n**4)
        + 3 * n * (5 + 3 * n) * (2 + 3 * n + n**2) ** 2 * S1**2
        - 2 * (2 + 3 * n + n**2) ** 3 * S1**3
        + 3 * n * (5 + 3 * n) * (2 + 3 * n + n**2) ** 2 * S2
        - 6
        * (2 + 3 * n + n**2)
        * S1
        * (n * (-9 - 8 * n + n**3) + (2 + 3 * n + n**2) ** 2 * S2)
        - 4 * (2 + 3 * n + n**2) ** 3 * S3
    ) / (n * (1 + n) ** 4 * (2 + n) ** 4)


@nb.njit(cache=True)
def lm14m2(n, S1, S2, S3, S4):
    r"""Mellin transform of :math:`(1-x)^2\ln^4(1-x)`.

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
        :math:`\mathcal{M}[(1-x)^2\ln^4(1-x)](N)`
    """
    return (
        2
        / (n * (1 + n) ** 5 * (2 + n) ** 5)
        * (
            +12 * n * (-33 + n * (-54 + n * (-15 + n * (20 + 3 * n * (5 + n)))))
            - 2 * n * (1 + n) ** 3 * (2 + n) ** 3 * (5 + 3 * n) * S1**3
            + (1 + n) ** 4 * (2 + n) ** 4 * S1**4
            + 6 * n * (1 + n) ** 2 * (2 + n) ** 2 * (-9 - 8 * n + n**3) * S2
            + 3 * (1 + n) ** 4 * (2 + n) ** 4 * S2**2
            + 6
            * (1 + n) ** 2
            * (2 + n) ** 2
            * S1**2
            * (n * (-9 - 8 * n + n**3) + (1 + n) ** 2 * (2 + n) ** 2 * S2)
            - 4 * n * (1 + n) ** 3 * (2 + n) ** 3 * (5 + 3 * n) * S3
            + 2
            * (1 + n)
            * (2 + n)
            * S1
            * (
                6 * n * (-17 + n * (-21 + 2 * n * (-1 + n * (3 + n))))
                - 3 * n * (1 + n) ** 2 * (2 + n) ** 2 * (5 + 3 * n) * S2
                + 4 * (1 + n) ** 3 * (2 + n) ** 3 * S3
            )
            + 6 * (1 + n) ** 4 * (2 + n) ** 4 * S4
        )
    )
