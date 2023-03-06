r"""
This module contains the |NLO| |OME| (OMEs) in the polarized case
for the matching conditions in the |VFNS|.
Heavy quark contribution for intrinsic evolution are not considered for the polarized case at this order.
The matching conditions for the |VFNS| at :math:`\mu_F^2 \neq m_H^2`
are provided in :cite:`Bierenbaum_2023`.
"""
import numba as nb
import numpy as np

from eko.constants import TR


@nb.njit(cache=True)
def A_hg(n, L):  # method 1 -> Following the un-polarized format in eko
    r"""
     |NLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(1)}` given in Eq. (104) of :cite:`Bierenbaum_2023`.

    Parameters
    ----------
        n : complex
            Mellin moment
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_hg : complex
            |NLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(1)}`
    """
    den = 1 / (n * (1 + n))
    num = 4 * (n - 1)
    return num * den * L * TR


@nb.njit(cache=True)
def A_hg(
    n, L
):  # method 2 -> following the Blumlein format without the - (method 3-> direct insertion of splitting funct from eko)
    r"""
    |NLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(1)}` given in Eq. (104) of :cite:`Bierenbaum_2023`.
    Parameters
    ----------
        n : complex
            Mellin moment
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_hg : complex
            |NLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(1)}`
    """
    DPqg0hat = (8 * TR * (n - 1)) / (n * (1 + n))
    return 1 / 2 * DPqg0hat * L


@nb.njit(cache=True)
def A_gg(L):  # method 1
    r"""
    |NLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(1)}` given in Eq. (186) of :cite:`Bierenbaum_2023`.

    Parameters
    ----------
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_gg : complex
            |NLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(1)}`
    """
    return -4.0 / 3.0 * L * TR


@nb.njit(cache=True)
def A_gg(L):  # method 2
    r"""
    |NLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(1)}` given in Eq. (186) of :cite:`Bierenbaum_2023`.

    Parameters
    ----------
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_gg : complex
            |NLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(1)}`
    """
    beta_0 = -4.0 / 3.0 * TR
    return beta_0 * L


@nb.njit(cache=True)
def A_singlet(n, L):
    r"""
    Computes the |NLO| singlet |OME|.
    .. math::
        A^{S,(1)} = \left(\begin{array}{cc}
        A_{gg,H}^{S,(1)} & 0 & 0\\
        0 & 0 & 0 \\
        A_{hg}^{S,(1)} & 0 & 0
        \end{array}\right)

    Parameters
    ----------
    n : complex
        Mellin moment
     L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
      -------
    A_S : numpy.ndarray
        |NLO| singlet |OME| :math:`A^{S,(1)}`

      See Also
      --------
        A_gg : :math:`A_{gg,H}^{S,(1)}`
        A_hg : :math:`A_{hg}^{S,(1)}`
    """
    A_S = np.array(
        [
            [A_gg(L), 0.0, 0.0],
            [0 + 0j, 0 + 0j, 0 + 0j],
            [A_hg(n, L), 0.0, 0.0],
        ],
        np.complex_,
    )
    return A_S
