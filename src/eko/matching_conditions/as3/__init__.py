# -*- coding: utf-8 -*-
r"""This module defines the matching conditions for the N3LO |VFNS| evolution.

The expressions are based on:

    - :cite:`Bierenbaum:2009mv`. Isabella Bierenbaum, Johannes Blümlein, and
      Sebastian Klein. Mellin Moments of the O(alpha**3(s)) Heavy Flavor
      Contributions to unpolarized Deep-Inelastic Scattering at Q**2 \ensuremath
      >\ensuremath > m**2 and Anomalous Dimensions. Nucl. Phys. B, 820:417-482,
      2009. arXiv:0904.3563, doi:10.1016/j.nuclphysb.2009.06.005.
    - :cite:`Bl_mlein_2000`. Johannes Blümlein. Analytic continuation of mellin
      transforms up to two-loop order. Computer Physics Communications,
      133(1):76-104, Dec 2000. URL:
      http://dx.doi.org/10.1016/S0010-4655(00)00156-9,
      doi:10.1016/s0010-4655(00)00156-9.
    - :cite:`Bierenbaum:2009zt`. Isabella Bierenbaum, Johannes Blümlein, and
      Sebastian Klein. The Gluonic Operator Matrix Elements at O(alpha(s)**2) for
      DIS Heavy Flavor Production. Phys. Lett. B, 672:401-406, 2009.
      arXiv:0901.0669, doi:10.1016/j.physletb.2009.01.057.
    - :cite:`Ablinger:2010ty`. J. Ablinger, J. Blümlein, S. Klein, C. Schneider,
      and F. Wissbrock. The $O(\alpha _s^3)$ Massive Operator Matrix Elements of
      $O(n_f)$ for the Structure Function $F_2(x,Q^2)$ and Transversity. Nucl.
      Phys. B, 844:26-54, 2011. arXiv:1008.3347,
      doi:10.1016/j.nuclphysb.2010.10.021.
    - :cite:`Ablinger:2014vwa`. J. Ablinger, A. Behring, J. Blümlein, A. De
      Freitas, A. Hasselhuhn, A. von Manteuffel, M. Round, C. Schneider, and F.
      Wißbrock. The 3-Loop Non-Singlet Heavy Flavor Contributions and Anomalous
      Dimensions for the Structure Function $F_2(x,Q^2)$ and Transversity. Nucl.
      Phys. B, 886:733-823, 2014. arXiv:1406.4654,
      doi:10.1016/j.nuclphysb.2014.07.010.
    - :cite:`Ablinger:2014uka`. J. Ablinger, J. Blümlein, A. De Freitas, A.
      Hasselhuhn, A. von Manteuffel, M. Round, and C. Schneider. The $O(\alpha
      _s^3 T_F^2)$ Contributions to the Gluonic Operator Matrix Element. Nucl.
      Phys. B, 885:280-317, 2014. arXiv:1405.4259,
      doi:10.1016/j.nuclphysb.2014.05.028.
    - :cite:`Behring:2014eya`. A. Behring, I. Bierenbaum, J. Blümlein, A. De
      Freitas, S. Klein, and F. Wißbrock. The logarithmic contributions to the
      $O(\alpha ^3_s)$ asymptotic massive Wilson coefficients and operator matrix
      elements in deeply inelastic scattering. Eur. Phys. J. C, 74(9):3033, 2014.
      arXiv:1403.6356, doi:10.1140/epjc/s10052-014-3033-x.
    - :cite:`Blumlein:2017wxd`. Johannes Blümlein, Jakob Ablinger, Arnd Behring,
      Abilio De Freitas, Andreas von Manteuffel, Carsten Schneider, and C.
      Schneider. Heavy Flavor Wilson Coefficients in Deep-Inelastic Scattering:
      Recent Results. PoS, QCDEV2017:031, 2017. arXiv:1711.07957,
      doi:10.22323/1.308.0031.
    - :cite:`Ablinger_2014`. J. Ablinger, J. Blümlein, A. De Freitas, A.
      Hasselhuhn, A. von Manteuffel, M. Round, C. Schneider, and F. Wißbrock. The
      transition matrix element a_gq(n) of the variable flavor number scheme at
      o(α_s^3). Nuclear Physics B, 882:263-288, May 2014. URL:
      http://dx.doi.org/10.1016/j.nuclphysb.2014.02.007,
      doi:10.1016/j.nuclphysb.2014.02.007.
    - :cite:`Ablinger_2015`. J. Ablinger, A. Behring, J. Blümlein, A. De
      Freitas, A. von Manteuffel, and C. Schneider. The 3-loop pure singlet heavy
      flavor contributions to the structure function f2(x,q2) and the anomalous
      dimension. Nuclear Physics B, 890:48-151, Jan 2015. URL:
      http://dx.doi.org/10.1016/j.nuclphysb.2014.10.008,
      doi:10.1016/j.nuclphysb.2014.10.008.
"""

import numba as nb
import numpy as np

from .agg import A_gg
from .agq import A_gq
from .aHg import A_Hg
from .aHq import A_Hq
from .aqg import A_qg
from .aqqNS import A_qqNS
from .aqqPS import A_qqPS


@nb.njit(cache=True)
def A_singlet(n, sx_singlet, sx_non_singlet, nf, L):
    r"""Computes the |N3LO| singlet |OME|.

    .. math::
        A^{S,(3)} = \left(\begin{array}{cc}
            A_{gg, H}^{S,(3)} & A_{gq, H}^{S,(3)} & 0 \\
            A_{qg, H}^{S,(3)} & A_{qq,H}^{NS,(3)} + A_{qq,H}^{PS,(3)} & 0\\
            A_{hg}^{S,(3)} & A_{hq}^{PS,(3)} & 0\\
        \end{array}\right)

    When using the code, please cite the complete list of references
    available at the top of this module :mod:`eko.matching_conditions.as3`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx_singlet : list
        singlet like harmonic sums cache containing:

        .. math ::
            [[S_1,S_{-1}],
            [S_2,S_{-2}],
            [S_{3}, S_{2,1}, S_{2,-1}, S_{-2,1}, S_{-2,-1}, S_{-3}],
            [S_{4}, S_{3,1}, S_{2,1,1}, S_{-2,-2}, S_{-3, 1}, S_{-4}],]

    sx_non_singlet: list
        same as sx_singlet but now for non-singlet like harmonics
    nf : int
        number of active flavor below the threshold
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    A_S : numpy.ndarray
        |NNLO| singlet |OME| :math:`A^{S,(3)}(N)`

    """
    A_hq_3 = A_Hq(n, sx_singlet, nf, L)
    A_hg_3 = A_Hg(n, sx_singlet, nf, L)

    A_gq_3 = A_gq(n, sx_singlet, nf, L)
    A_gg_3 = A_gg(n, sx_singlet, nf, L)

    A_qq_ps_3 = A_qqPS(n, sx_singlet, nf, L)
    A_qq_ns_3 = A_qqNS(n, sx_non_singlet, nf, L)
    A_qg_3 = A_qg(n, sx_singlet, nf, L)

    A_S = np.array(
        [
            [A_gg_3, A_gq_3, 0.0],
            [A_qg_3, A_qq_ps_3 + A_qq_ns_3, 0.0],
            [A_hg_3, A_hq_3, 0.0],
        ],
        np.complex_,
    )
    return A_S


@nb.njit(cache=True)
def A_ns(n, sx_all, nf, L):
    r"""Computes the |N3LO| non-singlet |OME|.

    .. math::
        A^{NS,(3)} = \left(\begin{array}{cc}
            A_{qq,H}^{NS,(3)} & 0\\
            0 & 0\\
        \end{array}\right)

    When using the code, please cite the complete list of references available
    at the top of this module :mod:`eko.matching_conditions.as3`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx_all : list
        harmonic sums cache containing:

        .. math ::
            [[S_1,S_{-1}],
            [S_2,S_{-2}],
            [S_{3}, S_{2,1}, S_{2,-1}, S_{-2,1}, S_{-2,-1}, S_{-3}],
            [S_{4}, S_{3,1}, S_{2,1,1}, S_{-2,-2}, S_{-3, 1}, S_{-4}],],
            [S_{5}, S_{-5}]

    nf : int
        number of active flavor below the threshold
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    A_NS : numpy.ndarray
        |N3LO| non-singlet |OME| :math:`A^{NS,(3)}`

    See Also
    --------
    A_qqNS_3 : :math:`A_{qq,H}^{NS,(3))}`

    """
    return np.array([[A_qqNS(n, sx_all, nf, L), 0.0], [0 + 0j, 0 + 0j]], np.complex_)
