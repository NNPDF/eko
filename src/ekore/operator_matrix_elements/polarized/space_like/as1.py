r"""The |NLO| |OME| in the polarized case for the matching conditions in the
|VFNS|.

Heavy quark contribution for intrinsic evolution are not considered for the polarized case.
The matching conditions for the |VFNS| at :math:`\mu_F^2 \neq m_H^2` are provided in :cite:`Bierenbaum:2022biv`.
In the paper, the fraction :math:`\mu_F^2 / m_H^2` inside the log is inverted, yielding an additional factor of (-1) wherever ``L`` has an odd power.
Additionally, a different convention for the anomalous dimensions is used, yielding a factor 2 in the |OME|'s wherever they are present.
The anomalous dimensions and beta function with the addition 'hat' have the form :math:`\hat\gamma = \gamma^{(nf+1)} - \gamma^{(nf)}`.
"""

import numba as nb
import numpy as np

from ....anomalous_dimensions.polarized.space_like.as1 import gamma_qg as gamma0_qg
from ...unpolarized.space_like.as1 import A_gg as A_gg_unpol


@nb.njit(cache=True)
def A_hg(n, L):
    r"""Compute the |NLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(1)}`.

    Implements :eqref:`104` of :cite:`Bierenbaum:2022biv`.

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
    complex
        |NLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(1)}`
    """
    gamma0_qghat = 2 * gamma0_qg(n, nf=1)
    return (1 / 2) * gamma0_qghat * (-L)


@nb.njit(cache=True)
def A_gg(L):
    r"""Compute the |NLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(1)}`.

    Implements :eqref:`186` of :cite:`Bierenbaum:2022biv`.

    Parameters
    ----------
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    complex
        |NLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(1)}`
    """
    return A_gg_unpol(L)


@nb.njit(cache=True)
def A_singlet(n, L):
    r"""Compute the |NLO| singlet |OME|.

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
    numpy.ndarray
        |NLO| singlet |OME| :math:`A^{S,(1)}`
    """
    A_S = np.array(
        [
            [A_gg(L), 0.0 + 0j, 0.0 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j],
            [A_hg(n, L), 0.0 + 0j, 0.0 + 0j],
        ],
        np.complex128,
    )
    return A_S
