# -*- coding: utf-8 -*-
import numba as nb
import numpy as np

# The expressions are based on:
# - :cite:`Bierenbaum:2009mv`. Isabella Bierenbaum, Johannes Blumlein, and Sebastian Klein. Mellin Moments of the O(alpha**3(s)) Heavy Flavor Contributions to unpolarized Deep-Inelastic Scattering at Q**2 \ensuremath >\ensuremath > m**2 and Anomalous Dimensions. Nucl. Phys. B, 820:417-482, 2009. arXiv:0904.3563, doi:10.1016/j.nuclphysb.2009.06.005. # pylint: disable=line-too-long
# - :cite:`Bl_mlein_2000`. Johannes Blümlein. Analytic continuation of mellin transforms up to two-loop order. Computer Physics Communications, 133(1):76-104, Dec 2000. URL: http://dx.doi.org/10.1016/S0010-4655(00)00156-9, doi:10.1016/s0010-4655(00)00156-9. # pylint: disable=line-too-long
# - :cite:`Bierenbaum:2009zt`. Isabella Bierenbaum, Johannes Blumlein, and Sebastian Klein. The Gluonic Operator Matrix Elements at O(alpha(s)**2) for DIS Heavy Flavor Production. Phys. Lett. B, 672:401-406, 2009. arXiv:0901.0669, doi:10.1016/j.physletb.2009.01.057. # pylint: disable=line-too-long
# - :cite:`Ablinger:2010ty`. J. Ablinger, J. Blumlein, S. Klein, C. Schneider, and F. Wissbrock. The $O(\alpha _s^3)$ Massive Operator Matrix Elements of $O(n_f)$ for the Structure Function $F_2(x,Q^2)$ and Transversity. Nucl. Phys. B, 844:26-54, 2011. arXiv:1008.3347, doi:10.1016/j.nuclphysb.2010.10.021. # pylint: disable=line-too-long
# - :cite:`Ablinger:2014vwa`. J. Ablinger, A. Behring, J. Blümlein, A. De Freitas, A. Hasselhuhn, A. von Manteuffel, M. Round, C. Schneider, and F. Wißbrock. The 3-Loop Non-Singlet Heavy Flavor Contributions and Anomalous Dimensions for the Structure Function $F_2(x,Q^2)$ and Transversity. Nucl. Phys. B, 886:733-823, 2014. arXiv:1406.4654, doi:10.1016/j.nuclphysb.2014.07.010. # pylint: disable=line-too-long
# - :cite:`Ablinger:2014uka`. J. Ablinger, J. Blümlein, A. De Freitas, A. Hasselhuhn, A. von Manteuffel, M. Round, and C. Schneider. The $O(\alpha _s^3 T_F^2)$ Contributions to the Gluonic Operator Matrix Element. Nucl. Phys. B, 885:280-317, 2014. arXiv:1405.4259, doi:10.1016/j.nuclphysb.2014.05.028. # pylint: disable=line-too-long
# - :cite:`Behring:2014eya`. A. Behring, I. Bierenbaum, J. Blümlein, A. De Freitas, S. Klein, and F. Wißbrock. The logarithmic contributions to the $O(\alpha ^3_s)$ asymptotic massive Wilson coefficients and operator matrix elements in deeply inelastic scattering. Eur. Phys. J. C, 74(9):3033, 2014. arXiv:1403.6356, doi:10.1140/epjc/s10052-014-3033-x. # pylint: disable=line-too-long
# - :cite:`Blumlein:2017wxd`. Johannes Blümlein, Jakob Ablinger, Arnd Behring, Abilio De Freitas, Andreas von Manteuffel, Carsten Schneider, and C. Schneider. Heavy Flavor Wilson Coefficients in Deep-Inelastic Scattering: Recent Results. PoS, QCDEV2017:031, 2017. arXiv:1711.07957, doi:10.22323/1.308.0031. # pylint: disable=line-too-long
# - :cite:`Ablinger_2014`. J. Ablinger, J. Blümlein, A. De Freitas, A. Hasselhuhn, A. von Manteuffel, M. Round, C. Schneider, and F. Wißbrock. The transition matrix element a_gq(n) of the variable flavor number scheme at o(α_s^3). Nuclear Physics B, 882:263-288, May 2014. URL: http://dx.doi.org/10.1016/j.nuclphysb.2014.02.007, doi:10.1016/j.nuclphysb.2014.02.007. # pylint: disable=line-too-long
# - :cite:`Ablinger_2015`. J. Ablinger, A. Behring, J. Blümlein, A. De Freitas, A. von Manteuffel, and C. Schneider. The 3-loop pure singlet heavy flavor contributions to the structure function f2(x,q2) and the anomalous dimension. Nuclear Physics B, 890:48-151, Jan 2015. URL: http://dx.doi.org/10.1016/j.nuclphysb.2014.10.008, doi:10.1016/j.nuclphysb.2014.10.008. # pylint: disable=line-too-long


@nb.njit("c16(c16,c16[:],u4,f8)", cache=True)
def A_qqPS_3(n, sx, nf, L):
    r"""
    Computes the |N3LO| singlet |OME| :math:`A_{qq}^{PS,(3)}(N)`.
    The expression is presented in :cite:`Bierenbaum:2009mv`

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : numpy.ndarray
            list S1 ... S5
        nf : int
            number of active flavor below the threshold
        L : float
            :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
        A_qqPS_3 : complex
            :math:`A_{qq}^{PS,(3)}(N)`
    """
    S1, S2, S3 = sx[0], sx[1], sx[2]
    a_qqPS_l0 = (
        0.3333333333333333
        * nf
        * (
            (
                -0.13168724279835392
                * (
                    -864.0
                    - 1008.0 * n
                    - 3408.0 * np.power(n, 2)
                    - 11704.0 * np.power(n, 3)
                    + 34274.0 * np.power(n, 4)
                    + 204541.0 * np.power(n, 5)
                    + 423970.0 * np.power(n, 6)
                    + 532664.0 * np.power(n, 7)
                    + 492456.0 * np.power(n, 8)
                    + 354532.0 * np.power(n, 9)
                    + 187681.0 * np.power(n, 10)
                    + 66389.0 * np.power(n, 11)
                    + 13931.0 * np.power(n, 12)
                    + 1330.0 * np.power(n, 13)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            + (
                0.3950617283950617
                * (
                    288.0
                    - 96.0 * n
                    - 928.0 * np.power(n, 2)
                    + 3272.0 * np.power(n, 3)
                    + 12030.0 * np.power(n, 4)
                    + 15396.0 * np.power(n, 5)
                    + 14606.0 * np.power(n, 6)
                    + 11454.0 * np.power(n, 7)
                    + 5937.0 * np.power(n, 8)
                    + 1744.0 * np.power(n, 9)
                    + 233.0 * np.power(n, 10)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                (
                    -96.0
                    - 16.0 * n
                    + 404.0 * np.power(n, 2)
                    + 452.0 * np.power(n, 3)
                    + 521.0 * np.power(n, 4)
                    + 430.0 * np.power(n, 5)
                    + 185.0 * np.power(n, 6)
                    + 40.0 * np.power(n, 7)
                )
                * (-0.5925925925925926 * np.power(S1, 2) - 0.5925925925925926 * S2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    34.19184080098401
                    + 2.962962962962963 * np.power(S1, 3)
                    + 8.88888888888889 * S1 * S2
                    + 5.925925925925926 * S3
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    a_qqPS_l3 = (1.1851851851851851 * np.power(2.0 + n + np.power(n, 2), 2) * nf) / (
        (-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n)
    )
    a_qqPS_l2 = (
        0.3333333333333333
        * nf
        * (
            (
                -3.5555555555555554
                * (
                    -24.0
                    - 20.0 * n
                    + 58.0 * np.power(n, 2)
                    + 61.0 * np.power(n, 3)
                    + 85.0 * np.power(n, 4)
                    + 83.0 * np.power(n, 5)
                    + 37.0 * np.power(n, 6)
                    + 8.0 * np.power(n, 7)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (10.666666666666666 * np.power(2.0 + n + np.power(n, 2), 2) * S1)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    a_qqPS_l1 = (
        0.3333333333333333
        * nf
        * (
            (
                1.1851851851851851
                * (
                    144.0
                    - 48.0 * n
                    - 808.0 * np.power(n, 2)
                    + 200.0 * np.power(n, 3)
                    + 3309.0 * np.power(n, 4)
                    + 4569.0 * np.power(n, 5)
                    + 4763.0 * np.power(n, 6)
                    + 4269.0 * np.power(n, 7)
                    + 2379.0 * np.power(n, 8)
                    + 712.0 * np.power(n, 9)
                    + 95.0 * np.power(n, 10)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            - (
                7.111111111111111
                * (
                    -24.0
                    - 20.0 * n
                    + 58.0 * np.power(n, 2)
                    + 61.0 * np.power(n, 3)
                    + 85.0 * np.power(n, 4)
                    + 83.0 * np.power(n, 5)
                    + 37.0 * np.power(n, 6)
                    + 8.0 * np.power(n, 7)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                1.0
                * np.power(2.0 + n + np.power(n, 2), 2)
                * (-10.666666666666666 * np.power(S1, 2) - 10.666666666666666 * S2)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    return a_qqPS_l0 + a_qqPS_l1 * L + a_qqPS_l2 * L**2 + a_qqPS_l3 * L**3
