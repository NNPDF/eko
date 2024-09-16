"""Auxilary functions for harmonics sums of weight = 3,4.

Implementations of some Mellin transformations :math:`g_k(N)` :cite:`MuselliPhD`
appearing in the analytic continuation of harmonics sums of weight = 3,4.
"""

import numba as nb
import numpy as np

from eko.constants import log2, zeta2, zeta3

from . import w1
from .polygamma import recursive_harmonic_sum as s

a1 = np.array(
    [
        0.999999974532238,
        -0.499995525889840,
        0.333203435557262,
        -0.248529457782640,
        0.191451164719161,
        -0.137466222728331,
        0.0792107412244877,
        -0.0301109656912626,
        0.00538406208663153,
        0.0000001349586745,
    ]
)

c1 = np.array(
    [
        2.2012182965269744e-8,
        2.833327652357064,
        -1.8330909624101532,
        0.7181879191200942,
        -0.0280403220046588,
        -0.181869786537805,
        0.532318519269331,
        -1.07281686995035,
        1.38194913357518,
        -1.11100841298484,
        0.506649587198046,
        -0.100672390783659,
    ]
)

c3 = np.array(
    [
        0,
        1.423616247405256,
        -0.08001203559240111,
        -0.39875367195395994,
        0.339241791547134,
        -0.0522116678353452,
        -0.0648354706049337,
        0.0644165053822532,
        -0.0394927322542075,
        0.0100879370657869,
    ]
)

p11 = np.array([11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0])
p32 = np.array([-25.0 / 24.0, 2.0, -3.0 / 2.0, 2.0 / 3.0, -1.0 / 8.0])
p31 = np.array([205.0 / 144.0, -25.0 / 12.0, 23.0 / 24.0, -13.0 / 36.0, 1.0 / 16])


@nb.njit(cache=True)
def mellin_g3(N, S1):
    r"""Compute the Mellin transform of :math:`\text{Li}_2(x)/(1+x)`.

    This function appears in the analytic continuation of the harmonic sum
    :math:`S_{-2,1}(N)` which in turn appears in the |NLO| anomalous dimension
    (see :ref:`theory/mellin:harmonic sums`).

    Parameters
    ----------
    N : complex
        Mellin moment
    S1: complex
            Harmonic sum :math:`S_{1}(N)`

    Returns
    -------
    mellin_g3 : complex
        approximate Mellin transform :math:`\mathcal{M}[\text{Li}_2(x)/(1+x)](N)`

    Note
    ----
        We use the name from :cite:`MuselliPhD`, but not his implementation - rather we use the
        Pegasus :cite:`Vogt:2004ns` implementation.
    """
    cs = [1.0000e0, -0.9992e0, 0.9851e0, -0.9005e0, 0.6621e0, -0.3174e0, 0.0699e0]
    g3 = 0
    for j, c in enumerate(cs):
        Nj = N + j
        g3 += c * (zeta2 - s(S1, N, j, 1) / Nj) / Nj
    return g3


@nb.njit(cache=True)
def mellin_g4(N):
    r"""Compute the Mellin transform of :math:`\text{Li}_2(-x)/(1+x)`.

    Implementation and definition in :eqref:`B.5.25` of :cite:`MuselliPhD` or
    in :eqref:`61` of :cite:`Bl_mlein_2000`, but none of them is fully correct.


    Parameters
    ----------
    N : complex
        Mellin moment

    Returns
    -------
    mellin_g4 : complex
        Mellin transform :math:`\mathcal{M}[\text{Li}_2(-x)/(1+x)](N)`
    """
    g4 = -1 / 2 * zeta2 * log2
    for k, ak in enumerate(a1):
        Nk = N + k + 1
        beta = 1 / 2 * (w1.S1((Nk) / 2) - w1.S1((Nk - 1) / 2))
        g4 += ak * (N / Nk * zeta2 / 2 + (k + 1) / Nk**2 * (log2 - beta))
    return g4


@nb.njit(cache=True)
def mellin_g5(N, S1, S2):
    r"""Compute the Mellin transform of :math:`(\text{Li}_2(x)ln(x))/(1+x)`.

    Implementation and definition in :eqref:`B.5.26` of :cite:`MuselliPhD` or
    in :eqref:`62` of :cite:`Bl_mlein_2000`, but none of them is fully correct.

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Harmonic sum :math:`S_{1}(N)`
        S2: complex
            Harmonic sum :math:`S_{2}(N)`

    Returns
    -------
        mellin_g5 : complex
            Mellin transform :math:`\mathcal{M}[(\text{Li}_2(x)ln(x))/(1+x)](N)`
    """
    g5 = 0.0
    for k, ak in enumerate(a1):
        Nk = N + k + 1
        poly1nk = -s(S2, N, k + 1, 2) + zeta2
        g5 -= ak * ((k + 1) / Nk**2 * (zeta2 + poly1nk - 2 * s(S1, N, k + 1, 1) / Nk))
    return g5


@nb.njit(cache=True)
def mellin_g6(N, S1):
    r"""Compute the Mellin transform of :math:`\text{Li}_3(x)/(1+x)`.

    Implementation and definition in :eqref:`B.5.27` of :cite:`MuselliPhD` or
    in :eqref:`63` of :cite:`Bl_mlein_2000`, but none of them is fully correct.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1: complex
        Harmonic sum :math:`S_{1}(N)`

    Returns
    -------
    mellin_g6 : complex
        Mellin transform :math:`\mathcal{M}[\text{Li}_3(x)/(1+x)](N)`
    """
    g6 = zeta3 * log2
    for k, ak in enumerate(a1):
        Nk = N + k + 1
        g6 -= ak * (
            N / Nk * zeta3 + (k + 1) / Nk**2 * (zeta2 - s(S1, N, k + 1, 1) / Nk)
        )
    return g6


@nb.njit(cache=True)
def mellin_g8(N, S1, S2):
    r"""Compute the Mellin transform of :math:`S_{1,2}(x)/(1+x)`.

    Implementation and definition in :eqref:`B.5.29` of :cite:`MuselliPhD` or
    in :eqref:`65` of :cite:`Bl_mlein_2000`, but none of them is fully correct.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1: complex
        Harmonic sum :math:`S_{1}(N)`
    S2: complex
        Harmonic sum :math:`S_{2}(N)`

    Returns
    -------
    mellin_g8 : complex
        Mellin transform :math:`\mathcal{M}[S_{1,2}(x)/(1+x)](N)`
    """
    g8 = zeta3 * log2
    for k, ak in enumerate(a1):
        Nk = N + k + 1
        g8 -= ak * (
            N / Nk * zeta3
            + (k + 1) / Nk**2 * 1 / 2 * (s(S1, N, k + 1, 1) ** 2 + s(S2, N, k + 1, 2))
        )
    return g8


@nb.njit(cache=True)
def mellin_g18(N, S1, S2):
    r"""Compute the Mellin transform of :math:`-(\text{Li}_2(x) - \zeta_2)/(1-x)`.

    Implementation and definition in :eqref:`124` of :cite:`Bl_mlein_2000`

    Note: comparing to :cite:`Bl_mlein_2000`, we believe :cite:`MuselliPhD`
    was not changing the notations of :math:`P^{(1)}_{2}` to :math:`P^{(1)}_{1}`.
    So we implement eq 124 of :cite:`Bl_mlein_2000` but using :cite:`MuselliPhD`
    notation.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`
    S2 : complex
        Harmonic sum :math:`S_{2}(N)`
    Returns
    -------
    mellin_g18 : complex
        Mellin transform :math:`\mathcal{M}[-(\text{Li}_2(x) - \zeta_2)/(1-x)](N)`

    """
    g18 = (S1**2 + S2) / (N) - zeta2 * S1
    for k, ck in enumerate(c1):
        Nk = N + k
        g18 += ck * (N) / (Nk) * s(S1, N, k, 1)
    for k, p11k in enumerate(p11):
        Nk = N + k
        g18 -= p11k * (N) / (Nk) * (s(S1, N, k, 1) ** 2 + s(S2, N, k, 2))
    return g18


@nb.njit(cache=True)
def mellin_g19(N, S1):
    r"""Compute the Mellin transform of :math:`-(\text{Li}_2(-x) +
    \zeta_2/2)/(1-x)`.

    Implementation and definition in :eqref:`B.5.40` of :cite:`MuselliPhD` or in :eqref:`125` of
    :cite:`Bl_mlein_2000`, but none of them is fully correct.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`

    Returns
    -------
    mellin_g19 : complex
        Mellin transform :math:`\mathcal{M}[-(\text{Li}_2(-x) + \zeta_2/2)/(1-x)](N)`
    """
    g19 = 1 / 2 * zeta2 * S1
    for k, ak in enumerate(a1):
        g19 -= ak / (k + 1) * s(S1, N, k + 1, 1)
    return g19


@nb.njit(cache=True)
def mellin_g21(N, S1, S2, S3):
    r"""Compute the Mellin transform of :math:`-(S_{1,2}(x) - \zeta_3)/(1-x)`.

    Implementation and definition in :eqref:`B.5.42 of` :cite:`MuselliPhD`.

    Note: comparing to :cite:`Bl_mlein_2000`, we believe :cite:`MuselliPhD`
    was not changing the notations of :math:`P^{(3)}_{2}` to :math:`P^{(3)}_{1}`
    and :math:`P^{(3)}_{3}` to :math:`P^{(3)}_{2}`.
    So we implement :eqref:`127` of :cite:`Bl_mlein_2000` but using :cite:`MuselliPhD`
    notation.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`
    S2 : complex
        Harmonic sum :math:`S_{2}(N)`
    S3 : complex
        Harmonic sum :math:`S_{3}(N)`

    Returns
    -------
    mellin_g21 : complex
        Mellin transform :math:`\mathcal{M}[-(S_{1,2}(x) - \zeta_3)/(1-x)](N)`

    """
    g21 = -zeta3 * S1 + (S1**3 + 3 * S1 * S2 + 2 * S3) / (2 * N)
    for k, ck in enumerate(c3):
        Nk = N + k
        g21 += ck * N / Nk * s(S1, N, k, 1)
    for k in range(0, 5):
        Nk = N + k
        S1nk = s(S1, N, k, 1)
        S2nk = s(S2, N, k, 2)
        S3nk = s(S3, N, k, 3)
        g21 += (
            N
            / Nk
            * (
                p32[k] * (S1nk**3 + 3 * S1nk * S2nk + 2 * S3nk)
                - p31[k] * (S1nk**2 + S2nk)
            )
        )
    return g21


@nb.njit(cache=True)
def mellin_g22(N, S1, S2, S3):
    r"""Compute the Mellin transform of :math:`-(\text{Li}_2(x) ln(x))/(1-x)`.

    Implementation and definition in :eqref:`B.5.43` of :cite:`MuselliPhD`.

    Note: comparing to :cite:`Bl_mlein_2000`, we believe :cite:`MuselliPhD`
    was not changing the notations of :math:`P^{(1)}_{2}` to :math:`P^{(1)}_{1}`
    So we implement :eqref:`128` of :cite:`Bl_mlein_2000` but using :cite:`MuselliPhD`
    notation.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1 : complex
        Harmonic sum :math:`S_{1}(N)`
    S2 : complex
        Harmonic sum :math:`S_{2}(N)`
    S3 : complex
        Harmonic sum :math:`S_{3}(N)`

    Returns
    -------
    mellin_g22 : complex
        Mellin transform :math:`\mathcal{M}[-(\text{Li}_2(x) ln(x))/(1-x)](N)`
    """
    g22 = 0.0
    for k, ck in enumerate(c1):
        poly1nk = -s(S2, N, k, 2) + zeta2
        g22 += ck * poly1nk
    for k, p11k in enumerate(p11):
        S1nk = s(S1, N, k, 1)
        poly1nk = -s(S2, N, k, 2) + zeta2
        poly2nk = 2 * s(S3, N, k, 3) + zeta3
        g22 -= p11k * (S1nk * poly1nk - 1 / 2 * poly2nk)
    return g22
