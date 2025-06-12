r"""Contains the |NNLO| |OME| in the polarized case for the matching conditions
in the |VFNS|.

The equations are given in :cite:`Bierenbaum:2022biv`. As in the |NLO|
|OME|, in the paper, an additional factor 2 can be found in front of the
anomalous dimensions and factor (-1) for odd powers of L. The anomalous
dimensions and beta function with the addition 'hat' are defined as in
the |NLO| case.
"""

import numba as nb
import numpy as np

from eko.constants import CA, CF, TR, zeta2, zeta3

from ....anomalous_dimensions.polarized.space_like.as1 import gamma_gq as gamma0_gq
from ....anomalous_dimensions.polarized.space_like.as1 import gamma_ns as gamma0_qq
from ....anomalous_dimensions.polarized.space_like.as1 import gamma_qg as gamma0_qg
from ....anomalous_dimensions.polarized.space_like.as2 import gamma_qg as gamma1_qg
from ....harmonics import cache as c

beta_0hat = -4 / 3 * TR
"""This is the lowest order beta function with the addition 'hat' defined as
above."""


@nb.njit(cache=True)
def A_qq_ns(n, cache, L):
    r"""Compute |NNLO| light-light non-singlet |OME| :math:`A_{qq,H}^{NS,(2)}`.

    Implements :eqref:`133` of :cite:`Bierenbaum:2022biv`.

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

    gamma1_qqNS = (
        -CF
        * TR
        * (4 / 3)
        * (
            -8 * S2
            + (40 / 3) * S1
            - (3 * n**4 + 6 * n**3 + 47 * n**2 + 20 * n - 12)
            / (3 * n**2 * (n + 1) ** 2)
        )
    )
    aqqns = (
        TR
        * CF
        * (
            -(8 / 3) * S3
            - (8 / 3) * zeta2 * S1
            + (40 / 9) * S2
            + 2 * zeta2 * ((3 * n**2 + 3 * n + 2) / (3 * n * (n + 1)))
            - (224 * S1 / 27)
            + (
                219 * n**6
                + 657 * n**5
                + 1193 * n**4
                + 763 * n**3
                - 40 * n**2
                - 48 * n
                + 72
            )
            / (54 * n**3 * (n + 1) ** 3)
        )
    )

    a_qq_l0 = aqqns - 1 / 4 * 2 * beta_0hat * gamma0_qq(n, cache) * zeta2
    a_qq_l1 = (1 / 2) * gamma1_qqNS
    a_qq_l2 = (1 / 4) * beta_0hat * 2 * gamma0_qq(n, cache)
    return a_qq_l2 * L**2 + a_qq_l1 * (-L) + a_qq_l0


@nb.njit(cache=True)
def A_hq_ps(n, cache, L, nf):
    r"""Compute |NNLO| heavy-light pure-singlet |OME| :math:`A_{Hq}^{PS,(2)}`.

    Implements :eqref:`138` of :cite:`Bierenbaum:2022biv`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`
    nf : int
        Number of active flavors

    Returns
    -------
    complex
        |NNLO| heavy-light pure-singlet |OME| :math:`A_{Hq}^{PS,(2)}`
    """
    S2 = c.get(c.S2, cache, n)

    a_hq_ps_l = (
        -4
        * TR
        * CF
        * ((n + 2) / (n**2 * (n + 1) ** 2))
        * (
            (n - 1) * (2 * S2 + zeta2)
            + ((4 * n**3 - 4 * n**2 - 3 * n - 1) / (n**2 * (n + 1) ** 2))
        )
    )
    # term that differentiates between M scheme and Larin scheme,
    # we are computing in M scheme hence the addition of this term
    z_qq_ps = -CF * TR * nf * ((8 * (2 + n) * (n**2 - n - 1)) / (n**3 * (n + 1) ** 3))
    gamma1_ps_qqhat = (16 * CF * (2 + n) * (1 + 2 * n + n**3) * TR) / (
        n**3 * ((1 + n) ** 3)
    )
    a_hq_l0 = (
        a_hq_ps_l
        + z_qq_ps
        + (zeta2 / 8) * (2 * gamma0_qg(n, nf=1)) * (2 * gamma0_gq(n))
    )
    a_hq_l1 = (1 / 2) * gamma1_ps_qqhat
    a_hq_l2 = -(1 / 8) * (2 * gamma0_qg(n, nf=1)) * (2 * gamma0_gq(n))
    return a_hq_l2 * L**2 + a_hq_l1 * (-L) + a_hq_l0


@nb.njit(cache=True)
def A_hg(n, cache, L):
    r"""Compute |NNLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(2)}`.

    Implements :eqref:`111` of :cite:`Bierenbaum:2022biv`.

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
    S3 = c.get(c.S3, cache, n)
    Sm21 = c.get(c.Sm21, cache, n, is_singlet=False)
    S2h = c.get(c.S2h, cache, n)
    S3h = c.get(c.S3h, cache, n)
    S2ph = c.get(c.S2ph, cache, n)
    S3ph = c.get(c.S3ph, cache, n)

    a_hg = (
        (1 / (6 * n**4 * (1 + n) ** 4 * (2 + n)))
        * TR
        * (
            -2
            * CF
            * (
                6
                * (
                    4
                    + 8 * n
                    + 3 * n**2
                    - 2 * n**3
                    - 25 * n**4
                    + 60 * n**6
                    + 52 * n**7
                    + 12 * n**8
                )
                + (-1 + n)
                * n**2
                * (1 + n) ** 2
                * (2 + n)
                * (2 + 3 * n + 3 * n**2)
                * np.pi**2
                + 12 * n**2 * (1 + n) ** 3 * (-36 - 22 * n - 2 * n**2 + n**3) * S1
                + 12 * n**2 * (1 + n) ** 3 * (-2 + 3 * n + 3 * n**2) * S1**2
                + 12
                * n**2
                * (1 + n) ** 2
                * (2 + 33 * n + 43 * n**2 + 17 * n**3 + n**4)
                * S2
                - 4
                * (-1 + n)
                * n**3
                * (1 + n) ** 3
                * (2 + n)
                * (S1**3 + S1 * (np.pi**2 + 3 * S2) - 4 * S3)
            )
            + CA
            * (
                -24
                * (
                    4
                    + 12 * n
                    + 33 * n**2
                    + 4 * n**3
                    + 29 * n**4
                    + 36 * n**5
                    + 22 * n**6
                    + 10 * n**7
                    + 2 * n**8
                )
                + 8 * (-1 + n) * n**2 * (1 + n) ** 2 * (2 + n) ** 2 * np.pi**2
                + 24 * n**3 * (1 + n) * (2 - 10 * n - n**2 + 4 * n**3 + n**4) * S1
                + 24 * n**3 * (1 + n) ** 2 * (5 + 4 * n + n**2) * S1**2
                + 24 * n**2 * (1 + n) ** 2 * (-16 + 15 * n + 24 * n**2 + 7 * n**3) * S2
                - 24
                * (1 - n)
                * n**3
                * (2 + n)
                * (4 + (1 + n) ** 2 * S2h - (1 + n) ** 2 * S2ph)
                + (1 - n)
                * n**3
                * (2 + n)
                * (
                    -48
                    + 8 * (1 + n) ** 3 * S1**3
                    + 72 * (1 + n) ** 3 * S1 * S2
                    - 24 * (1 + n) * S1 * (4 + (1 + n) ** 2 * S2h - (1 + n) ** 2 * S2ph)
                    + 64 * (1 + n) ** 3 * S3
                    + 6 * (1 + n) ** 3 * S3ph
                    - 6 * (1 + n) ** 3 * (S3h - zeta3)
                    + 18 * (1 + n) ** 3 * zeta3
                    - 12 * (1 + n) ** 3 * (8 * Sm21 + 5 * zeta3)
                )
            )
        )
    )

    # remove the nf dependence from
    # (2 * gamma0_gg(n, S1, nf) + 2 * beta_0(nf))
    pgg0_beta0 = 8 * CA * (-(2 / (n + n**2)) + S1)
    a_hg_l0 = a_hg + (1 / 8) * 2 * gamma0_qg(n, nf=1) * (
        pgg0_beta0 - 2 * gamma0_qq(n, cache)
    )
    a_hg_l1 = 2 * gamma1_qg(n, nf=1, cache=cache)
    a_hg_l2 = (
        (1 / 8)
        * 2
        * gamma0_qg(n, nf=1)
        * (2 * gamma0_qq(n, cache) - pgg0_beta0 - 4 * beta_0hat)
    )
    return a_hg_l2 * L**2 + a_hg_l1 * (-L) + a_hg_l0


@nb.njit(cache=True)
def A_gq(n, cache, L):
    r"""Compute |NNLO| gluon-quark |OME| :math:`A_{gq,H}^{S,(2)}`.

    Implements :eqref:`174` of :cite:`Bierenbaum:2022biv`.

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

    gamma1_gq_hat = 2 * (
        16
        * CF
        * TR
        * (2 + n)
        * ((2 + 5 * n) / (9 * n * (1 + n) ** 2) - S1 / (3 * n * (1 + n)))
    )
    a_gq_l0 = (
        CF
        * TR
        * (
            (n + 2)
            * (
                8 * (22 + 41 * n + 28 * n**2) / (27 * n * (n + 1) ** 3)
                - 8 * (2 + 5 * n) / (9 * n * (n + 1) ** 2) * S1
                + ((S1**2 + S2) * 4) / (3 * n * (n + 1))
            )
        )
    )
    # here there is minus sing missing in the paper,
    # passing from eq 172 to 174
    a_gq_l1 = gamma1_gq_hat / 2
    a_gq_l2 = beta_0hat * gamma0_gq(n)

    return a_gq_l2 * L**2 + a_gq_l1 * (-L) + a_gq_l0


@nb.njit(cache=True)
def A_gg(n, cache, L):
    r"""Compute |NNLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(2)}`.

    Implements :eqref:`187` of :cite:`Bierenbaum:2022biv`.

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
    ggg1_canf = (
        -5 * S1 / 9
        + (-3 + 13 * n + 16 * n**2 + 6 * n**3 + 3 * n**4) / (9 * n**2 * (1 + n) ** 2)
    ) * 4
    ggg1_cfnf = (4 + 2 * n - 8 * n**2 + n**3 + 5 * n**4 + 3 * n**5 + n**6) / (
        n**3 * (1 + n) ** 3
    )
    gamma1_gg_hat = 8 * TR * (CA * ggg1_canf + CF * ggg1_cfnf)

    a_gg_f = (
        -15 * n**8
        - 60 * n**7
        - 82 * n**6
        - 44 * n**5
        - 15 * n**4
        - 4 * n**2
        - 12 * n
        - 8
    ) / (n**4 * (1 + n) ** 4)
    a_gg_a = (
        2 * (15 * n**6 + 45 * n**5 + 374 * n**4 + 601 * n**3 + 161 * n**2 - 24 * n + 36)
    ) / (27 * n**3 * (1 + n) ** 3) - (4 * S1 * (47 + 56 * n) / (27 * (1 + n)))
    a_gg_l0 = TR * (CF * a_gg_f + CA * a_gg_a)

    # here there is minus sing missing in the paper,
    # passing from eq 177 to 185
    a_gg_l1 = gamma1_gg_hat / 2

    # remove the nf dependence from
    # (2 * gamma0_gg(n, S1, nf) + 2 * beta_0(nf))
    pgg0_beta0 = 8 * CA * (-(2 / (n + n**2)) + S1)
    a_gg_l2 = (1 / 8) * (
        2 * beta_0hat * pgg0_beta0
        + 2 * gamma0_gq(n) * 2 * gamma0_qg(n, nf=1)
        + 8 * beta_0hat**2
    )
    return a_gg_l2 * L**2 + a_gg_l1 * (-L) + a_gg_l0


@nb.njit(cache=True)
def A_singlet(n, cache, L, nf):
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
    nf : int
        Number of active flavors

    Returns
    -------
    numpy.ndarray
        |NNLO| singlet |OME| :math:`A^{S,(2)}(N)`
    """
    return np.array(
        [
            [A_gg(n, cache, L), A_gq(n, cache, L), 0.0],
            [0.0, A_qq_ns(n, cache, L), 0.0],
            [A_hg(n, cache, L), A_hq_ps(n, cache, L, nf), 0.0],
        ],
        np.complex128,
    )


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
