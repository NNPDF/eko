r"""
This module contains the |NLO| |OME| in the polarized case for the matching conditions in the |VFNS|.

Heavy quark contribution for intrinsic evolution are not considered for the polarized case.
The matching conditions for the |VFNS| at :math:`\mu_F^2 \neq m_H^2` are provided in :cite:`Bierenbaum_2023`. In the paper, the fraction :math:`\mu_F^2 / m_H^2` inside the log is inverted, yielding an additional factor of (-1) wherever L has an odd power. Additionally, a different convention for the anomalous dimensions is used, yielding a factor 2 in the |OME|'s wherever they are present. The anomalous dimensions and beta function with the addition 'hat', have the form :math:`\gamma_hat = gamma(nf+1) - gamma(nf)`.
"""
import numba as nb
import numpy as np

from eko.constants import TR

from ....anomalous_dimensions.polarized.space_like.as1 import gamma_qg as gamma0_qg
from ...unpolarized.space_like.as1 import A_gg as A_gg_unpol


@nb.njit(cache=True)
def gamma0_qghat(n, nf):
    r"""Compute the |LO| polarized quark-gluon anomalous dimension with the addition 'hat' and thus, discarding the part proportional 'nf'. The factor 2 is included due to convention mentioned above.

    Parameters
    ----------
    N : complex
      Mellin moment
    nf : int
      Number of active flavors

    Returns
    -------
    complex
      |LO| polarized quark-gluon anomalous dimension (hat) :math:`\\\hat{gamma_{qg}}^{(0)}(N)`

    """
    return 2 * gamma0_qg(n, nf) / nf


@nb.njit(cache=True)
def A_hg(n, L, nf):
    r"""
    |NLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(1)}` given in Eq. (104) of :cite:`Bierenbaum_2023`.
    Parameters
    ----------
        n : complex
            Mellin moment
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`
        nf : int
            Number of active flavors

    Returns
    -------
        A_hg : complex
            |NLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(1)}`
    """
    return (1 / 2) * gamma0_qghat(n, nf) * (-L)


@nb.njit(cache=True)
def A_gg(L):
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
    return 2 * A_gg_unpol(L)


@nb.njit(cache=True)
def A_singlet(n, L, nf):
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
    nf : int
        Number of active flavors

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
            [A_hg(n, L, nf), 0.0, 0.0],
        ],
        np.complex_,
    )
    return A_S
