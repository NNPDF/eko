r"""
This module contains the |NNLO| OME's in the polarized case for the matching conditions in the |VFNS|.
The equations are given in :cite:`Bierenbaum_2023`. As in the |NLO| OME's, in the paper, an additional factor 2 can be found in front of the anomalous dimensions and factor (-1) for odd powers of L. The anomalous dimensions and beta function with the addition 'hat' are defined as in the |NLO| case.


"""
import numba as nb
import numpy as np

from eko import constants

from eko.beta import beta_qcd_as2 as beta_0
from ....anomalous_dimensions.polarized.space_like.as1 import gamma_gg as gamma0_gg
from ....anomalous_dimensions.polarized.space_like.as1 import gamma_gq as gamma0_gq
from ....anomalous_dimensions.polarized.space_like.as1 import gamma_qg as gamma0_qg
from ....anomalous_dimensions.unpolarized.space_like.as1 import gamma_ns as gamma0_qq
from ....harmonics.constants import zeta2, zeta3
from ....harmonics.polygamma import cern_polygamma
from ...polarized.space_like.as1 import gamma0_qghat

beta_0hat = (
    -4 / 3
) * constants.TR  # this is the lowest order beta function with the addition 'hat' defined as above


@nb.njit(cache=True)
def A_qq_ns(n, sx, L):
    r"""
    |NNLO| light-light non-singlet |OME| :math:`A_{qq,H}^{NS,(2)}` given in
    Eq. (133) in :cite:`Bierenbaum_2023`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_qq_ns : complex
            |NNLO| light-light non-singlet |OME| :math:`A_{qq,H}^{NS,(2)}`
    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]

    gamma1_qqNS = (
        -constants.CF
        * constants.TR
        * (4 / 3)
        * (
            -8 * S2
            + (40 / 3) * S1
            - (3 * n**4 + 6 * n**3 + 47 * n**2 + 20 * n - 12)
            / (3 * n**2 * (n + 1) ** 2)
        )
    )  # relative to the paper there is a factor -1
    aqqns = (
        constants.TR
        * constants.CF
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

    a_qq_l0 = aqqns - (1 / 4) * beta_0hat * gamma0_qq(n, S1) * zeta2
    a_qq_l1 = (1 / 2) * gamma1_qqNS
    a_qq_l2 = (1 / 4) * beta_0hat * 2 * gamma0_qq(n, S1)
    return a_qq_l2 * L**2 + a_qq_l1 * (-L) + a_qq_l0


@nb.njit(cache=True)
def A_hq_ps(n, sx, L, nf):
    r"""
    |NNLO| heavy-light pure-singlet |OME| :math:`A_{Hq}^{PS,(2)}` given in
    Eq. (138) in :cite:`Bierenbaum_2023`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`
        nf : int
            Number of active flavors

    Returns
    -------
        A_hq_ps : complex
            |NNLO| heavy-light pure-singlet |OME| :math:`A_{Hq}^{PS,(2)}`
    """
    S2 = sx[1][1]

    a_hq_ps_l = (
        -4
        * constants.TR
        * constants.CF
        * ((n + 2) / (n**2 * (n + 1) ** 2))(
            (n - 1) * (2 * S2 + zeta2)
            - ((4 * n**3 - 4 * n**2 - 3 * n - 1) / (n**2 * (n + 1) ** 2))
        )
    )
    z_qq_ps = (
        -constants.CF
        * constants.TR
        * nf
        * ((8 * (2 + n) * (n**2 - n - 1)) / (n**3 * (n + 1) ** 3))
    )  # term that differentiates between M scheme and Larin scheme, we are computing in M scheme hence the addition of this term

    gamma1_ps_qqhat = (
        16 * constants.CF * (2 + n) * (1 + 2 * n + n**3) * constants.TR
    ) / (n**3 * ((1 + n) ** 3))

    a_hq_l0 = (
        a_hq_ps_l + z_qq_ps + (zeta2 / 8) * (gamma0_qghat(n, nf)) * (2 * gamma0_gq(n))
    )
    a_hq_l1 = (1 / 2) * gamma1_ps_qqhat
    a_hq_l2 = -(1 / 8) * (gamma0_qghat(n, nf)) * (2 * gamma0_gq(n))
    return a_hq_l2 * L**2 + a_hq_l1 * (-L) + a_hq_l0


@nb.njit(cache=True)
def A_hg(n, sx, L, nf):
    r"""
    |NNLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(2)}` given in
    Eq. (111) in :cite:`Bierenbaum_2023`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`
        nf : int
            Number of active flavors

    Returns
    -------
        A_hg : complex
            |NNLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(2)}`
    """
    S1 = sx[0][0]
    S2, Sm2 = sx[1]
    S3, Sm21 = sx[2]

    a_hg = constants.CF * constants.TR * (
        4
        * (n - 1)
        / (3 * n * (n + 1))(-4 * S3 + S1**3 + 3 * S1 * S2 + 6 * S1 * zeta2)
        - 4
        * (n**4 + 17 * n**3 + 43 * n**2 + 33 * n + 2)
        / ((n**2) * ((n + 1) ** 2) * (n + 2))
        * S2
        - 4 * (3 * n**2 + 3 * n - 2) / ((n**2) * (n + 1) * (n + 2)) * (S1**2)
        - 2
        * (n - 1)
        * (3 * (n**3) + 3 * n + 2)
        / (
            (n**2) * ((n + 1) ** 2) * zeta2
            - 4
            * ((n**3) - 2 * (n**2) - 22 * n - 36)
            / ((n**2) * (n + 1) * (n + 2))
            * S1
            - (
                12 * (n**8)
                + 52 * (n**7)
                + 60 * (n**6)
                - 25 * (n**4)
                - 2 * (n**3)
                + 3 * (n**2)
                + 8 * n
                + 4
            )
            / ((n**4) * ((n + 1) ** 4) * (n + 2))
        )
    ) + constants.TR * constants.CA * (
        (4 * (n - 1) / (3 * n(n + 1)))
        * (
            (
                12 * (-1) ** (n + 1) * (Sm21 + 5 / 8 * zeta3)
            )  # regarding Felix's comment here i can either make it a function for possibilities of (-1) ** (n + 1) or write it as a seperate variable
            + 3
            * (1 / 2)
            * (
                (-1 / 4) * cern_polygamma(n / 2, 2)
                + (1 / 4) * cern_polygamma((1 + n) / 2, 2)
            )
            - 8 * S3
            - S1**3
            - 9 * S1 * S2
            - 12
            * S1
            * (1 / 2)
            * (
                (-1 / 2) * cern_polygamma(n / 2, 1)
                + (1 / 2) * cern_polygamma((1 + n) / 2, 1)
            )
            - 3 * zeta3
        )
        - 16
        * (1 / 2)
        * (
            (-1 / 2) * cern_polygamma(n / 2, 1)
            + (1 / 2) * cern_polygamma((1 + n) / 2, 1)
        )((n - 1) / (n * (n + 1) ** 2))
        + (4 * (n**2 + 4 * n + 5) * S1**2) / (n * (n + 1) ** 2 * (n + 2))
        + 4
        * S2
        * (7 * n**3 + 24 * n**2 + 15 * n - 16)
        / (n**2 * (n + 1) ** 2 * (n + 2))
        + 8 * (n - 1) * (n + 2) * zeta2 / (n**2 * (n + 1) ** 2) * zeta2
        + 4
        * S1
        * (n**4 + 4 * n**3 - n**2 - 10 * n + 2)
        / (n * (n + 1) ** 3 * (n + 2))
        - 4
        * (
            2 * n**8
            + 10 * n**7
            + 22 * n**6
            + 36 * n**5
            + 29 * n**4
            + 4 * n**3
            + 33 * n**2
            + 12 * n
            + 4
        )
        / (n**4 * (n + 1) * (n + 2))
    )
    gamma1_qg_hat = -constants.CF * constants.TR * (
        (8 * (n - 1) * (2 - n + 10 * n**3 + 5 * n**4)) / ((n**3) * (1 + n) ** 3)
        - (32 * (-1 + n) * S1) / (n**2 * (1 + n))
        + (16 * (-1 + n)(S1**2 - S2)) / (n * (1 + n))
    ) - constants.CA * constants.TR * (
        (16 * (-2 - 7 * n + 3 * n**2 - 4 * n**3 + n**4 + n**5))
        / (n**3 * (1 + n) ** 3)
        + (64 * S1) / (n * (1 + n) ** 2)
        - (16 * (-1 + n) * (2 * Sm2 + S1**2 + S2)) / (n * (1 + n))
    )  # relative to the paper there is a factor -1

    a_hg_l0 = a_hg + (1 / 8) * 2 * gamma0_qg(n, nf) * (
        2 * gamma0_gg(n, S1, nf) - 2 * gamma0_qq(n, S1) + 2 * beta_0(nf)
    )
    a_hg_l1 = gamma1_qg_hat * (1 / 2)
    a_hg_l2 = (1 / 8) * (
        2 * gamma0_qq(n, S1) - 2 * gamma0_gg(n, S1, nf) - 2 * beta_0(nf) - 4 * beta_0hat
    )

    return a_hg_l2 * L**2 + a_hg_l1 * (-L) + a_hg_l0


@nb.njit(cache=True)
def A_gq(n, sx, L):
    r"""
    |NNLO| gluon-quark |OME| :math:`A_{gq,H}^{S,(2)}` given in
    Eq. (174) in :cite:`Bierenbaum_2023`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_gq : complex
            |NNLO| gluon-quark |OME| :math:`A_{gq,H}^{S,(2)}`
    """
    S1 = sx[0][0]
    S2 = sx[1][0]

    gamma1_gq_hat = -2 * (
        16
        * constants.CF
        * (
            ((2 + n) * (2 + 5 * n)) / (9 * n * (1 + n) ** 2)
            - ((2 + n) * S1) / (3 * n * (1 + n))
        )
        * constants.TR
    )

    a_gq_l0 = (
        constants.CF
        * constants.TR
        * (
            (n + 2)
            * (
                8 * (22 + 41 * n + 28 * n**2) / (27 * n * (n + 1) ** 3)
                - 8 * (2 + 5 * n) / (9 * n * (n + 1) ** 2) * S1
                + ((S1**2 + S2) * 4) / (3 * n * (n + 1))
            )
        )
    )
    a_gq_l1 = -(1 / 2) * gamma1_gq_hat
    a_gq_l2 = -(beta_0hat / 2) * (2 * gamma0_gq(n))

    return a_gq_l2 * L**2 + a_gq_l1 * (-L) + a_gq_l0


@nb.njit(cache=True)
def A_gg(n, sx, L, nf):
    r"""
    |NNLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(2)} ` given in
    Eq. (187) in :cite:`Bierenbaum_2023`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`
        nf : int
            Number of active flavors

    Returns
    -------
        A_gg : complex
            |NNLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(2)}`
    """
    S1 = sx[0][0]
    gamma1_gg_hat = 8 * constants.TR(
        -(
            (
                constants.CF(
                    4 + 2 * n - 8 * n**2 + n**3 + 5 * n**4 + 3 * n**5 + n**6
                )
            )
            / ((n**3) * (1 + n) ** 3)
        )
        - 4
        * constants.CA
        * (
            (-3 + 13 * n + 16 * (n**2) + 6 * (n**3) + 3 * (n**4))
            / (9 * (n**2) * (1 + n) ** 2)
            - (5 * S1) / 9
        )
    )

    a_gg_f = (
        -15 * (n**8)
        - 60 * (n**7)
        - 82 * (n**6)
        - 44 * (n**5)
        - 15 * (n**4)
        - 4 * (n**2)
        - 12 * n
        - 8
    ) / ((n**4) * (1 + n) ** 4)
    a_gg_a = (
        2
        * (
            15 * n**6
            + 45 * n**5
            + 374 * n**4
            + 601 * n**3
            + 161 * n**2
            - 24 * n
            + 36
        )
    ) / (27 * (n**3) * ((1 + n) ** 3)) - (4 * S1 * (47 + 56 * n) / (27 * (1 + n)))

    a_gg_l0 = constants.TR * (constants.CF * a_gg_f + constants.CA * a_gg_a)
    a_gg_l1 = (1 / 2) * gamma1_gg_hat
    a_gg_l2 = (1 / 8) * (
        2 * beta_0hat * (-2 * gamma0_gg(n, S1, nf) + 2 * beta_0(nf))
        + 2 * gamma0_gq(n) * gamma0_qghat(n, nf)
        + 8 * beta_0hat**2
    )

    return a_gg_l2 * L**2 + a_gg_l1 * (-L) + a_gg_l0


@nb.njit(cache=True)
def A_singlet(
    n, sx, L, nf
):  # for future larin constant contribution could be a chosen parameter if useful distinction in some way
    r"""
      Computes the |NNLO| singlet |OME|.

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
        sx : list
            harmonic sums cache containing:
                [[:math:`S_1,S_{-1}`],[:math:`S_2,S_{-2}`],[:math:`S_3,S_{-2,1},S_{-3}`]]
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`
        nf : int
            Number of active flavors

      Returns
      -------
        A_S : numpy.ndarray
            |NNLO| singlet |OME| :math:`A^{S,(2)}(N)`
    """
    return np.array(
        [
            [A_gg(n, sx, L, nf), A_gq(n, sx, L), 0.0],
            [0.0, A_qq_ns(n, sx, L), 0.0],
            [A_hg(n, sx, L, nf), A_hq_ps(n, sx, L, nf), 0.0],
        ],
        np.complex_,
    )


@nb.njit(cache=True)
def A_ns(n, sx, L):
    r"""
    Computes the |NNLO| non-singlet |OME|.
      .. math::
          A^{NS,(2)} = \left(\begin{array}{cc}
            A_{qq,H}^{NS,(2)} & 0 \\
            0 & 0 \\
          \end{array}\right)

      Parameters
      ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache containing:
                [[:math:`S_1,S_{-1}`],[:math:`S_2,S_{-2}`],[:math:`S_3,S_{-2,1},S_{-3}`]]
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

      Returns
      -------
        A_NS : numpy.ndarray
            |NNLO| non-singlet |OME| :math:`A^{NS,(2)}`

      See Also
      --------
        A_qq_ns : :math:`A_{qq,H}^{NS,(2)}`
    """
    return np.array([[A_qq_ns(n, sx, L), 0.0], [0 + 0j, 0 + 0j]], np.complex_)