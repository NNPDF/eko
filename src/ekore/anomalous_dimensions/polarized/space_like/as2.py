"""The |NLO| Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from eko.constants import CA, CF, TR, zeta2, zeta3

from ....harmonics import cache as c

# Non Singlet sector is swapped
from ...unpolarized.space_like.as2 import gamma_nsm as gamma_nsp
from ...unpolarized.space_like.as2 import gamma_nsp as gamma_nsm  # noqa


@nb.njit(cache=True)
def gamma_ps(n, nf):
    r"""Compute the |NLO| polarized pure-singlet quark-quark anomalous dimension
    :cite:`Gluck:1995yr` (eq A.3).

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        number of active flavors

    Returns
    -------
    complex
        |NLO| pure-singlet quark-quark anomalous dimension :math:`\\gamma_{ps}^{(1)}(n)`
    """
    gqqps1_nfcf = (2 * (n + 2) * (1 + 2 * n + n**3)) / ((1 + n) ** 3 * n**3)
    result = 4.0 * TR * nf * CF * gqqps1_nfcf
    return result


@nb.njit(cache=True)
def gamma_qg(n, nf, cache):
    r"""Compute the |NLO| polarized quark-gluon singlet anomalous dimension
    :cite:`Gluck:1995yr` (eq A.4).

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NLO| quark-gluon singlet anomalous dimension :math:`\\gamma_{qg}^{(1)}(n)`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    Sp2m = c.get(c.S2mh, cache, n)
    gqg1_nfca = (
        (S1**2 - S2 + Sp2m) * (n - 1) / (n * (n + 1))
        - 4 * S1 / (n * (1 + n) ** 2)
        - (-2 - 7 * n + 3 * n**2 - 4 * n**3 + n**4 + n**5) / (n**3 * (1 + n) ** 3)
    ) * 2.0
    gqg1_nfcf = (
        (-(S1**2) + S2 + 2 * S1 / n) * (n - 1) / (n * (n + 1))
        - (n - 1)
        * (1 + 3.5 * n + 4 * n**2 + 5 * n**3 + 2.5 * n**4)
        / (n**3 * (1 + n) ** 3)
        + 4 * (n - 1) / (n**2 * (1 + n) ** 2)
    ) * 2
    result = 4.0 * TR * nf * (CA * gqg1_nfca + CF * gqg1_nfcf)
    return result


@nb.njit(cache=True)
def gamma_gq(n, nf, cache):
    r"""Compute the |NLO| polarized gluon-quark singlet anomalous dimension
    :cite:`Gluck:1995yr` (eq A.5).

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NLO| gluon-quark singlet anomalous dimension :math:`\\gamma_{gq}^{(1)}(n)`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    Sp2m = c.get(c.S2mh, cache, n)
    ggq1_cfcf = (
        (2 * (S1**2 + S2) * (n + 2)) / (n * (n + 1))
        - (2 * S1 * (n + 2) * (1 + 3 * n)) / (n * (1 + n) ** 2)
        - ((n + 2) * (2 + 15 * n + 8 * n**2 - 12.0 * n**3 - 9.0 * n**4))
        / (n**3 * (1 + n) ** 3)
        + 8 * (n + 2) / (n**2 * (1 + n) ** 2)
    ) * 0.5
    ggq1_cfca = -(
        -(-(S1**2) - S2 + Sp2m) * (n + 2) / (n * (n + 1))
        - S1 * (12 + 22 * n + 11 * n**2) / (3 * n**2 * (n + 1))
        + (36 + 72 * n + 41 * n**2 + 254 * n**3 + 271 * n**4 + 76 * n**5)
        / (9 * n**3 * (1 + n) ** 3)
    )
    ggq1_cfnf = (-S1 * (n + 2)) / (3 * n * (n + 1)) + ((n + 2) * (2 + 5 * n)) / (
        9 * n * (1 + n) ** 2
    )
    result = 4 * CF * (CA * ggq1_cfca + CF * ggq1_cfcf + 4.0 * TR * nf * ggq1_cfnf)
    return result


@nb.njit(cache=True)
def gamma_gg(n, nf, cache):
    r"""Compute the |NLO| polarized gluon-gluon singlet anomalous dimension
    :cite:`Gluck:1995yr` (eq A.6).

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NLO| gluon-quark singlet anomalous dimension :math:`\\gamma_{gq}^{(1)}(n)`
    """
    S1 = c.get(c.S1, cache, n)
    Sp1m = c.get(c.S1mh, cache, n)
    Sp2m = c.get(c.S2mh, cache, n)
    Sp3m = c.get(c.S3mh, cache, n)
    S1h = c.get(c.S1h, cache, n)
    g3 = c.get(c.g3, cache, n)
    SSCHLM = zeta2 / 2 * (+Sp1m - S1h + 2 / n) - S1 / n**2 - g3 - 5 * zeta3 / 8
    ggg1_caca = (
        -4 * S1 * Sp2m
        - Sp3m
        + 8 * SSCHLM
        + 8 * Sp2m / (n * (n + 1))
        + 2.0
        * S1
        * (72 + 144 * n + 67 * n**2 + 134 * n**3 + 67 * n**4)
        / (9 * n**2 * (n + 1) ** 2)
        - (144 + 258 * n + 7 * n**2 + 698 * n**3 + 469 * n**4 + 144 * n**5 + 48 * n**6)
        / (9 * n**3 * (1 + n) ** 3)
    ) * 0.5
    ggg1_canf = (
        -5 * S1 / 9
        + (-3 + 13 * n + 16 * n**2 + 6 * n**3 + 3 * n**4) / (9 * n**2 * (1 + n) ** 2)
    ) * 4
    ggg1_cfnf = (4 + 2 * n - 8 * n**2 + n**3 + 5 * n**4 + 3 * n**5 + n**6) / (
        n**3 * (1 + n) ** 3
    )
    result = 4 * (CA**2 * ggg1_caca + TR * nf * (CA * ggg1_canf + CF * ggg1_cfnf))
    return result


@nb.njit(cache=True)
def gamma_singlet(n, nf, cache):
    r"""Compute the |NLO| polarized singlet anomalous dimension matrix.

        .. math::
            \gamma_S^{(1)} = \left(\begin{array}{cc}
            \gamma_{qq}^{(1)} & \gamma_{qg}^{(1)}\\
            \gamma_{gq}^{(1)} & \gamma_{gg}^{(1)}
            \end{array}\right)

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    numpy.ndarray
        |NLO| singlet anomalous dimension matrix :math:`\gamma_{S}^{(1)}(N)`
    """
    gamma_qq = gamma_nsp(n, nf, cache) + gamma_ps(n, nf)
    gamma_S_0 = np.array(
        [
            [gamma_qq, gamma_qg(n, nf, cache)],
            [gamma_gq(n, nf, cache), gamma_gg(n, nf, cache)],
        ],
        np.complex128,
    )
    return gamma_S_0
