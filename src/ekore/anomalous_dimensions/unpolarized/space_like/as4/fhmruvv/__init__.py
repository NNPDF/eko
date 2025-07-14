"""The |FHMRUVV| |N3LO| Altarelli-Parisi splitting kernels approximations.

Authors follow Pegasus convention and so there is an additional global
minus sign with respect to our conventions.
"""

import numba as nb
import numpy as np

from .ggg import gamma_gg
from .ggq import gamma_gq
from .gnsm import gamma_nsm
from .gnsp import gamma_nsp
from .gnsv import gamma_nsv
from .gps import gamma_ps
from .gqg import gamma_qg


@nb.njit(cache=True)
def gamma_singlet(N, nf, cache, variation):
    r"""Compute the |N3LO| singlet anomalous dimension matrix.

      .. math::
          \gamma_S^{(3)} = \left(\begin{array}{cc}
          \gamma_{qq}^{(3)} & \gamma_{qg}^{(3)}\\
          \gamma_{gq}^{(3)} & \gamma_{gg}^{(3)}
          \end{array}\right)

    Parameters
    ----------
    N : complex
      Mellin moment
    nf : int
      Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache
    variation : tuple
        |N3LO| anomalous dimension variation ``(gg, gq, qg, qq)``

    Returns
    -------
    numpy.ndarray
        |N3LO| singlet anomalous dimension matrix
        :math:`\gamma_{S}^{(3)}(N)`
    """
    gamma_qq = gamma_nsp(N, nf, cache, variation[3]) + gamma_ps(
        N, nf, cache, variation[3]
    )
    gamma_S_0 = np.array(
        [
            [gamma_qq, gamma_qg(N, nf, cache, variation[2])],
            [
                gamma_gq(N, nf, cache, variation[1]),
                gamma_gg(N, nf, cache, variation[0]),
            ],
        ],
        np.complex128,
    )
    return gamma_S_0


@nb.njit(cache=True)
def gamma_singlet_qed(N, nf, cache, variation):
    r"""Compute the leading-order singlet anomalous dimension matrix for the
    unified evolution basis.

    .. math::
        \\gamma_S^{(3,0)} = \\left(\begin{array}{cccc}
        \\gamma_{gg}^{(3,0)} & 0 & \\gamma_{gq}^{(3,0)} & 0\\
        0 & 0 & 0 & 0 \\
        \\gamma_{qg}^{(3,0)} & 0 & \\gamma_{qq}^{(3,0)} & 0 \\
        0 & 0 & 0 & \\gamma_{qq}^{(3,0)} \\
        \\end{array}\right)

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache
    variation : tuple
        |N3LO| anomalous dimension variation ``(gg, gq, qg, qq)``

    Returns
    -------
    numpy.ndarray
        Leading-order singlet anomalous dimension matrix :math:`\\gamma_{S}^{(3,0)}(N)`
    """
    gamma_np_p = gamma_nsp(N, nf, cache, variation[3])
    gamma_qq = gamma_np_p + gamma_ps(N, nf, cache, variation[3])
    gamma_S = np.array(
        [
            [
                gamma_gg(N, nf, cache, variation[0]),
                0.0 + 0.0j,
                gamma_gq(N, nf, cache, variation[1]),
                0.0 + 0.0j,
            ],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [gamma_qg(N, nf, cache, variation[2]), 0.0 + 0.0j, gamma_qq, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, gamma_np_p],
        ],
        np.complex128,
    )
    return gamma_S


@nb.njit(cache=True)
def gamma_valence_qed(N, nf, cache, variation):
    r"""Compute the leading-order valence anomalous dimension matrix for the
    unified evolution basis.

    .. math::
        \\gamma_V^{(3,0)} = \\left(\begin{array}{cc}
        \\gamma_{nsV}^{(3,0)} & 0\\
        0 & \\gamma_{ns-}^{(3,0)}
        \\end{array}\right)

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache
    variation : tuple
        |N3LO| anomalous dimension variation ``(nsm, nsv)``

    Returns
    -------
    numpy.ndarray
        Leading-order singlet anomalous dimension matrix :math:`\\gamma_{V}^{(3,0)}(N)`
    """
    gamma_V = np.array(
        [
            [gamma_nsv(N, nf, cache, variation[-1]), 0.0],
            [0.0, gamma_nsm(N, nf, cache, variation[-2])],
        ],
        np.complex128,
    )
    return gamma_V
