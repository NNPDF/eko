r"""The unpolarized, space-like anomalous dimension
:math:`\gamma_{ns,+}^{(3)}`."""

import numba as nb

from eko.constants import CF, zeta3

from ......harmonics import cache as c
from ......harmonics.log_functions import lm11, lm11m1, lm12m1, lm13m1


@nb.njit(cache=True)
def gamma_nsp(n, nf, cache, variation):
    r"""Compute the |N3LO| singlet-like non-singlet anomalous dimension.

    The routine is taken from :cite:`Moch:2017uml`.

    The :math:`nf^{0,1}` leading large-nc contributions and the :math:`nf^2` part
    are high-accuracy (0.1% or better) parametrizations of the exact
    results. The :math:`nf^3` expression is exact up to numerical truncations.

    The remaining :math:`nf^{0,1}` terms are approximations based on the first
    eight even moments together with small-x and large-x constraints.
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
        |N3LO| singlet-like non-singlet anomalous dimension
        :math:`\gamma_{ns,+}^{(3)}(N)`
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
    P3NPA01 = (
        -(107.16 / n**7)
        + 339.753 / n**6
        - 1341.01 / n**5
        + 2412.94 / n**4
        - 3678.88 / n**3
        - 2118.87 * lm11(n, S1)
        - 1777.27 * lm12m1(n, S1, S2)
        - 204.183 * lm13m1(n, S1, S2, S3)
        + 1853.56 / n
        - 8877.38 / (1 + n)
        + 7393.83 / (2 + n)
        - 2464.61 / (3 + n)
    )
    P3NPA02 = (
        -(107.16 / n**7)
        + 339.753 / n**6
        - 1341.01 / n**5
        + 379.152 / n**3
        - 1389.73 / n**2
        - 2118.87 * lm11(n, S1)
        - 173.936 * lm12m1(n, S1, S2)
        + 223.078 * lm13m1(n, S1, S2, S3)
        - (2096.54 / n)
        + 8698.39 / (1 + n)
        - 19188.9 / (2 + n)
        + 10490.5 / (3 + n)
    )

    P3NPA11 = (
        -(33.5802 / n**6)
        + 111.802 / n**5
        + 50.772 / n**4
        - 118.608 / n**3
        + 337.931 * lm11(n, S1)
        - 143.813 * lm11m1(n, S1)
        - 18.8803 * lm13m1(n, S1, S2, S3)
        + 304.82503 / n
        - 1116.34 / (1 + n)
        + 2187.58 / (2 + n)
        - 1071.24 / (3 + n)
    )
    P3NPA12 = (
        -(33.5802 / n**6)
        + 111.802 / n**5
        - 204.341 / n**4
        + 267.404 / n**3
        + 337.931 * lm11(n, S1)
        - 745.573 * lm11m1(n, S1)
        + 8.61438 * lm13m1(n, S1, S2, S3)
        - (385.52331999999996 / n)
        + 690.151 / (1 + n)
        - 656.386 / (2 + n)
        + 656.386 / (3 + n)
    )

    # nf^2 (parametrized) and nf^3 (exact)
    P3NSPA2 = -(
        -193.85906555742952
        - 18.962964 / n**5
        + 99.1605 / n**4
        - 225.141 / n**3
        + 393.0056000000001 / n**2
        - 403.50217685814835 / n
        - 34.425000000000004 / (1 + n) ** 4
        + 108.42 / (1 + n) ** 3
        - 93.8225 / (1 + n) ** 2
        + 534.725 / (1 + n)
        + 246.50250000000003 / (2 + n)
        - 25.455 / ((1 + n) ** 2 * (2 + n))
        - (16.97 * n) / ((1 + n) ** 2 * (2 + n))
        + 8.485 / ((1 + n) * (2 + n))
        - 110.015 / (3 + n)
        + 78.9875 / (4 + n)
        + 195.5772 * S1
        - (101.0775 * S1) / n**2
        + (35.17361 * S1) / n
        - (8.485 * S1) / (1 + n)
        - (101.0775 * S2) / n
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
    P3NSPAI = P3NSA0 + nf * P3NSA1 + nf**2 * P3NSPA2 + nf**3 * P3NSA3
    if variation == 1:
        P3NSPA = P3NSPAI + P3NPA01 + nf * P3NPA11
    elif variation == 2:
        P3NSPA = P3NSPAI + P3NPA02 + nf * P3NPA12
    else:
        P3NSPA = P3NSPAI + 0.5 * ((P3NPA01 + P3NPA02) + nf * (P3NPA11 + P3NPA12))

    # The singular piece.
    A4qI = (
        2.120902 * 10**4 - 5.179372 * 10**3 * nf
        # + 1.955772 * 10**2 * nf**2
        # + 3.272344 * nf**3
    )
    A4ap1 = -507.152 + 7.33927 * nf
    A4ap2 = -505.209 + 7.53662 * nf
    D1 = 1 / n - S1
    if variation == 1:
        P3NSPB = (A4qI + A4ap1) * D1
    elif variation == 2:
        P3NSPB = (A4qI + A4ap2) * D1
    else:
        P3NSPB = (A4qI + 0.5 * (A4ap1 + A4ap2)) * D1

    # ..The local piece.
    B4qI = (
        2.579609 * 10**4 + 0.08 - (5.818637 * 10**3 + 0.97) * nf
        # + (1.938554 * 10**2 + 0.0037) * nf**2
        # + 3.014982 * nf**3
    )
    B4ap1 = -2405.03 + 267.965 * nf
    B4ap2 = -2394.47 + 269.028 * nf
    if variation == 1:
        P3NSPC = B4qI + B4ap1
    elif variation == 2:
        P3NSPC = B4qI + B4ap2
    else:
        P3NSPC = +B4qI + 0.5 * (B4ap1 + B4ap2)

    return -(P3NSPA + P3NSPB + P3NSPC)
