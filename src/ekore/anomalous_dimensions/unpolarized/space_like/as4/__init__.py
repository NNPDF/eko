"""The |N3LO| Altarelli-Parisi splitting kernels.

For further documentation see :doc:`N3LO anomalous dimensions <../../../theory/N3LO_ad>`

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
        |N3LO| anomalous dimension variation ``(gg_var, gq_var, qg_var, qq_var)``

    Returns
    -------
    numpy.ndarray
        |N3LO| singlet anomalous dimension matrix
        :math:`\gamma_{S}^{(3)}(N)`

    """
    gamma_qq = gamma_nsp(N, nf, cache) + gamma_ps(N, nf, cache, variation[3])
    gamma_S_0 = np.array(
        [
            [gamma_qq, gamma_qg(N, nf, cache, variation[2])],
            [
                gamma_gq(N, nf, cache, variation[1]),
                gamma_gg(N, nf, cache, variation[0]),
            ],
        ],
        np.complex_,
    )
    return gamma_S_0
