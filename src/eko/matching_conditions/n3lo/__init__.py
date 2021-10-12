# -*- coding: utf-8 -*-
"""
This module defines the matching conditions for the N3LO |VFNS| evolution.
"""
# import numba as nb
import numpy as np

from .agg import A_gg_3
from .agq import A_gq_3
from .aqg import A_qg_3
from .aqqPS import A_qqPS_3
from .aqqNS import A_qqNS_3
from .aHg import A_Hg_3
from .aHq import A_Hq_3


# @nb.njit("c16[:,:](c16,c16[:],u4)", cache=True)
def A_singlet_3(n, sx_all, nf):
    r"""
      Computes the |N3LO| singlet |OME|.

      .. math::
          A^{S,(3)} = \left(\begin{array}{cc}
            A_{gg, H}^{S,(3)} & A_{gq, H}^{S,(3)} & 0
            A_{qg, H}^{S,(3)} & A_{qq,H}^{NS,(3)} + A_{qq,H}^{PS,(3)} & 0\\
            A_{hg}^{S,(3)} & A_{hq}^{PS,(3)} & 0\\
          \end{array}\right)

      Parameters
      ----------
        n : complex
            Mellin moment
        sx_all : numpy.ndarray
            List of harmonic sums containing:
                [S1 ... S5, Sm1 ... Sm5, S21, S2m1, Sm21, Sm2m1, S31, S221, Sm22, Sm211, Sm31]

      Returns
      -------
        A_S_3 : numpy.ndarray
            |NNLO| singlet |OME| :math:`A^{S,(3)}(N)`
    """
    sx = sx_all[:5]
    smx = sx_all[5:10]
    s3x = sx_all[10:14]
    s4x = sx_all[14:]
    A_hq = A_Hq_3(n, sx, smx, s3x, s4x, nf)
    A_hg = A_Hg_3(n, sx, smx, s3x, s4x, nf)

    A_gq = A_gq_3(n, sx, smx, s3x, s4x, nf)
    A_gg = A_gg_3(n, sx, smx, s3x, s4x, nf)

    A_qq_ps = A_qqPS_3(n, sx, nf)
    A_qq_ns = A_qqNS_3(n, sx, smx, s3x, s4x, nf)
    A_qg = A_qg_3(n, sx, smx, s3x, s4x, nf)

    A_S_3 = np.array(
        [[A_gg, A_gq, 0.0], [A_qg, A_qq_ps + A_qq_ns, 0.0], [A_hg, A_hq, 0.0]],
        np.complex_,
    )
    return A_S_3


# @nb.njit("c16[:,:](c16,c16[:],u4)", cache=True)
def A_ns_3(n, sx_all, nf):
    r"""
      Computes the |N3LO| non-singlet |OME|.

      .. math::
          A^{NS,(3)} = \left(\begin{array}{cc}
            A_{qq,H}^{NS,(3)} & 0\\
            0 & 0\\
          \end{array}\right)

      Parameters
      ----------
        n : complex
            Mellin moment
        sx_all : numpy.ndarray
            List of harmonic sums containing:
                [S1 ... S5, Sm1 ... Sm5, S21, S2m1, Sm21, Sm2m1, S31, S221, Sm22, Sm211, Sm31]

      Returns
      -------
        A_NS_3 : numpy.ndarray
            |N3LO| non-singlet |OME| :math:`A^{NS,(3)}`

      See Also
      --------
        A_qqNS_3 : :math:`A_{qq,H}^{NS,(3))}`
    """
    sx = sx_all[:5]
    smx = sx_all[5:10]
    s3x = sx_all[10:14]
    s4x = sx_all[14:]
    A_qq = A_qqNS_3(n, sx, smx, s3x, s4x, nf)
    return np.array([[A_qq, 0.0], [0 + 0j, 0 + 0j]], np.complex_)
