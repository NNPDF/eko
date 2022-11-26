"""This subpackage contains the |N3LO| Altarelli-Parisi splitting kernels.

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
def gamma_singlet(N, nf, sx):
    r"""Computes the |N3LO| singlet anomalous dimension matrix

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
    sx : list
      harmonic sums cache

    Returns
    -------
    numpy.ndarray
        |N3LO| singlet anomalous dimension matrix
        :math:`\gamma_{S}^{(3)}(N)`

    See Also
    --------
    gamma_nsp : :math:`\gamma_{ns,+}^{(3)}`
    gamma_ps : :math:`\gamma_{ps}^{(3)}`
    gamma_qg : :math:`\gamma_{qg}^{(3)}`
    gamma_gq : :math:`\gamma_{gq}^{(3)}`
    gamma_gg : :math:`\gamma_{gg}^{(3)}`

    """
    gamma_qq = gamma_nsp(N, nf, sx) + gamma_ps(N, nf, sx)
    gamma_S_0 = np.array(
        [[gamma_qq, gamma_qg(N, nf, sx)], [gamma_gq(N, nf, sx), gamma_gg(N, nf, sx)]],
        np.complex_,
    )
    return gamma_S_0
