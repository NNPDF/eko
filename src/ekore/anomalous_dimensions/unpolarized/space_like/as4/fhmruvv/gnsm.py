r"""The unpolarized, space-like anomalous dimension
:math:`\gamma_{ns,-}^{(3)}`."""

import numba as nb

from eko.constants import CF, zeta3

from ......harmonics import cache as c
from ......harmonics.log_functions import lm11, lm11m1, lm12m1, lm13m1


@nb.njit(cache=True)
def gamma_nsm(n, nf, cache, variation):
    r"""Compute the |N3LO| valence-like non-singlet anomalous dimension.

    The routine is taken from :cite:`Moch:2017uml`.

    The :math:`nf^{0,1}` leading large-nc contributions and the :math:`nf^2` part are
    high-accuracy (0.1% or better) parametrizations of the exact
    results. The :math:`nf^3` expression is exact up to numerical truncations.

    The remaining :math:`nf^{0,1}` terms are approximations based on the first
    eight odd moments together with small-x and large-x constraints.
    The two sets spanning the error estimate are called via  IMOD = 1
    and  IMOD = 2.  Any other value of IMOD invokes their average.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| valence-like non-singlet anomalous dimension
        :math:`\gamma_{ns,-}^{(3)}(N)`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)

    # Leading large-n_c, nf^0 and nf^1, parametrized
    P3NSA0 = (
        360.0 / n**7
        - 1920.0 / n**6
        + 7147.812 / n**5
        - 17179.356 / n**4
        + 34241.9 / n**3
        - 51671.329999999994 / n**2
        + 19069.8 * lm11(n, S1)
        - (491664.8019540468 / n)
        - 4533.0 / (1 + n) ** 3
        - 11825.0 / (1 + n) ** 2
        + 129203.0 / (1 + n)
        - 254965.0 / (2 + n)
        + 83377.5 / (3 + n)
        - 45750.0 / (4 + n)
        + (49150.0 * (6.803662258392675 + n) * S1) / (n**2 * (1.0 + n))
        + (334400.0 * S2) / n
    )
    P3NSA1 = (
        160.0 / n**6
        - 864.0 / n**5
        + 2583.1848 / n**4
        - 5834.624 / n**3
        + 9239.374 / n**2
        - 3079.76 * lm11(n, S1)
        - (114047.0 / n)
        - 465.0 / (1 + n) ** 4
        - 1230.0 / (1 + n) ** 3
        + 7522.5 / (1 + n) ** 2
        + 55669.3 / (1 + n)
        - 43057.8 / (2 + n)
        + 13803.8 / (3 + n)
        - 7896.0 / (4 + n)
        - (120.0 * (-525.063 + n) * S1) / (n**2 * (1.0 + n))
        + (63007.5 * S2) / n
    )

    # Nonleading large-n_c, nf^0 and nf^1: two approximations
    P3NMA01 = (
        0.4964335 * (720 / n**7 - 720.0 / n**6)
        - 13.5288 / n**4
        + 1618.07 / n**2
        - 2118.8669999999997 * lm11(n, S1)
        + 31897.8 * lm11m1(n, S1)
        + 4653.76 * lm12m1(n, S1, S2)
        + 3902.3590000000004 / n
        + 5992.88 / (1 + n)
        + 19335.7 / (2 + n)
        - 31321.4 / (3 + n)
    )
    P3NMA02 = (
        +0.4964335 * (720 / n**7 - 2160.0 / n**6)
        - 189.6138 / n**4
        + 3065.92 / n**3
        - 2118.8669999999997 * lm11(n, S1)
        - 3997.39 * lm11m1(n, S1)
        + 511.567 * lm13m1(n, S1, S2, S3)
        - (2099.268 / n)
        + 4043.59 / (1 + n)
        - 19430.190000000002 / (2 + n)
        + 15386.6 / (3 + n)
    )

    P3NMA11 = (
        +64.7083 / n**5
        - 254.024 / n**3
        + 337.931 * lm11(n, S1)
        + 1856.63 * lm11m1(n, S1)
        + 440.17 * lm12m1(n, S1, S2)
        + 419.53485 / n
        + 114.457 / (1 + n)
        + 2341.816 / (2 + n)
        - 2570.73 / (3 + n)
    )

    P3NMA12 = (
        -17.0616 / n**6
        - 19.53254 / n**3
        + 337.931 * lm11(n, S1)
        - 1360.04 * lm11m1(n, S1)
        + 38.7337 * lm13m1(n, S1, S2, S3)
        - (367.64646999999997 / n)
        + 335.995 / (1 + n)
        - 1269.915 / (2 + n)
        + 1605.91 / (3 + n)
    )

    # nf^2 (parametrized) and nf^3 (exact)
    P3NSMA2 = -(
        -193.84583328013258
        - 23.7037032 / n**5
        + 117.5967 / n**4
        - 256.5896 / n**3
        + 437.881 / n**2
        + 720.385709813466 / n
        - 48.720000000000006 / (1 + n) ** 4
        + 189.51000000000002 / (1 + n) ** 3
        + 391.02500000000003 / (1 + n) ** 2
        + 367.4750000000001 / (1 + n)
        + 404.47249999999997 / (2 + n)
        - 2063.325 / ((1 + n) ** 2 * (2 + n))
        - (1375.55 * n) / ((1 + n) ** 2 * (2 + n))
        + 687.775 / ((1 + n) * (2 + n))
        - 81.71999999999998 / (3 + n)
        + 114.9225 / (4 + n)
        + 195.5772 * S1
        - (817.725 * S1) / n**2
        + (714.46361 * S1) / n
        - (687.775 * S1) / (1 + n)
        - (817.725 * S2) / n
    )
    eta = 1 / n * 1 / (n + 1)
    P3NSA3 = -CF * (
        -32 / 27 * zeta3 * eta
        - 16 / 9 * zeta3
        - 16 / 27 * eta**4
        - 16 / 81 * eta**3
        + 80 / 27 * eta**2
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

    # Assembly regular piece.
    P3NSMAI = P3NSA0 + nf * P3NSA1 + nf**3 * P3NSA3 + nf**2 * P3NSMA2
    if variation == 1:
        P3NSMA = P3NSMAI + P3NMA01 + nf * P3NMA11
    elif variation == 2:
        P3NSMA = P3NSMAI + P3NMA02 + nf * P3NMA12
    else:
        P3NSMA = P3NSMAI + 0.5 * ((P3NMA01 + P3NMA02) + nf * (P3NMA11 + P3NMA12))

    # The singular piece.
    A4qI = (
        2.120902 * 10**4 - 5.179372 * 10**3 * nf
        # + 1.955772 * 10**2 * nf**2
        # + 3.272344 * nf**3
    )
    A4ap1 = -511.228 + 7.08645 * nf
    A4ap2 = -502.481 + 7.82077 * nf
    D1 = 1 / n - S1
    if variation == 1:
        P3NSMB = (A4qI + A4ap1) * D1
    elif variation == 2:
        P3NSMB = (A4qI + A4ap2) * D1
    else:
        P3NSMB = (A4qI + 0.5 * (A4ap1 + A4ap2)) * D1

    # The local piece.
    B4qI = (
        2.579609 * 10**4 + 0.08 - (5.818637 * 10**3 + 0.97) * nf
        # + (1.938554 * 10**2 + 0.0037) * nf**2
        # + 3.014982 * nf**3
    )
    B4ap1 = -2426.05 + 266.674 * nf - 0.05 * nf
    B4ap2 = -2380.255 + 270.518 * nf - 0.05 * nf
    if variation == 1:
        P3NSMC = B4qI + B4ap1
    elif variation == 2:
        P3NSMC = B4qI + B4ap2
    else:
        P3NSMC = +B4qI + 0.5 * (B4ap1 + B4ap2)

    return -(P3NSMA + P3NSMB + P3NSMC)
