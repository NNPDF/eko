# -*- coding: utf-8 -*-
"""
    This file contains the next-to-leading-order Altarelli-Parisi splitting kernels.
"""

import numpy as np
import numba as nb

from eko import ekomath


@nb.njit
def gamma_nsm_1(n, nf: int, CA: float, CF: float):
    """
        Computes the next-to-leading-order valence-like non-singlet anomalous dimension.

        Implements Eq. (3.5) of :cite:`Moch:2004pa`.

        Parameters
        ----------
            n : complex
                Mellin moment
            nf : int
                Number of active flavours
            CA : float
                Casimir constant of adjoint representation
            CF : float
                Casimir constant of fundamental representation

        Returns
        -------
            gamma_nsm_1 : complex
                Next-to-leading-order valence-like non-singlet anomalous dimension
                :math:`\\gamma_{ns-}^{(1)}(N)`
    """
    S1 = ekomath.harmonic_S1(n)
    S2 = ekomath.harmonic_S2(n)
    # Here, Sp refers to S' ("s-prime") (german: "s-strich" or in Pegasus language: SSTR)
    # of :cite:`Gluck:1989ze` and NOT to the Spence function a.k.a. dilogarithm
    Sp1m = ekomath.harmonic_S1((n - 1) / 2)
    Sp2m = ekomath.harmonic_S2((n - 1) / 2)
    Sp3m = ekomath.harmonic_S3((n - 1) / 2)
    g3n = ekomath.mellin_g3(n)
    zeta2 = ekomath.zeta2
    zeta3 = ekomath.zeta3
    # fmt: off
    gqq1m_cfca = 16*g3n - (144 + n*(1 + n)*(156 + n*(340 + n*(655 + 51*n*(2 + n)))))/(18.*np.power(n,3)*np.power(1 + n,3)) + (-14.666666666666666 + 8/n - 8/(1 + n))*S2 - (4*Sp2m)/(n + np.power(n,2)) + S1*(29.77777777777778 + 16/np.power(n,2) - 16*S2 + 8*Sp2m) + 2*Sp3m + 10*zeta3 + zeta2*(16*S1 - 16*Sp1m - (16*(1 + n*np.log(2)))/n) # pylint: disable=line-too-long
    gqq1m_cfcf = -32*g3n + (24 - n*(-32 + 3*n*(-8 + n*(3 + n)*(3 + np.power(n,2)))))/(2.*np.power(n,3)*np.power(1 + n,3)) + (12 - 8/n + 8/(1 + n))*S2 + S1*(-24/np.power(n,2) - 8/np.power(1 + n,2) + 16*S2 - 16*Sp2m) + (8*Sp2m)/(n + np.power(n,2)) - 4*Sp3m - 20*zeta3 + zeta2*(-32*S1 + 32*Sp1m + 32*(1/n + np.log(2))) # pylint: disable=line-too-long
    gqq1m_cfnf = (-12 + n*(20 + n*(47 + 3*n*(2 + n))))/(9.*np.power(n,2)*np.power(1 + n,2)) - (40*S1)/9. + (8*S2)/3. # pylint: disable=line-too-long
    # fmt: on
    result = CF * CA * gqq1m_cfca + CF * CF * gqq1m_cfcf + CF * nf * gqq1m_cfnf
    return result


@nb.njit
def gamma_nsp_1(n, nf: int, CA: float, CF: float):
    """
        Computes the next-to-leading-order singlet-like non-singlet anomalous dimension.

        Implements Eq. (3.5) of :cite:`Moch:2004pa`.

        Parameters
        ----------
            n : complex
                Mellin moment
            nf : int
                Number of active flavours
            CA : float
                Casimir constant of adjoint representation
            CF : float
                Casimir constant of fundamental representation

        Returns
        -------
            gamma_nsm_1 : complex
                Next-to-leading-order singlet-like non-singlet anomalous dimension
                :math:`\\gamma_{ns+}^{(1)}(N)`
    """
    S1 = ekomath.harmonic_S1(n)
    S2 = ekomath.harmonic_S2(n)
    # Here, Sp refers to S' ("s-prime") (german: "s-strich" or in Pegasus language: SSTR)
    # of :cite:`Gluck:1989ze` and NOT to the Spence function a.k.a. dilogarithm
    Sp1p = ekomath.harmonic_S1(n / 2)
    Sp2p = ekomath.harmonic_S2(n / 2)
    Sp3p = ekomath.harmonic_S3(n / 2)
    g3n = ekomath.mellin_g3(n)
    zeta2 = ekomath.zeta2
    zeta3 = ekomath.zeta3
    # fmt: off
    gqq1p_cfca = -16*g3n + (132 - n*(340 + n*(655 + 51*n*(2 + n))))/(18.*np.power(n,2)*np.power(1 + n,2)) + (-14.666666666666666 + 8/n - 8/(1 + n))*S2 - (4*Sp2p)/(n + np.power(n,2)) + S1*(29.77777777777778 - 16/np.power(n,2) - 16*S2 + 8*Sp2p) + 2*Sp3p + 10*zeta3 + zeta2*(16*S1 - 16*Sp1p + 16*(1/n - np.log(2))) # pylint: disable=line-too-long
    gqq1p_cfcf = 32*g3n - (8 + n*(32 + n*(40 + 3*n*(3 + n)*(3 + np.power(n,2)))))/(2.*np.power(n,3)*np.power(1 + n,3)) + (12 - 8/n + 8/(1 + n))*S2 + S1*(40/np.power(n,2) - 8/np.power(1 + n,2) + 16*S2 - 16*Sp2p) + (8*Sp2p)/(n + np.power(n,2)) - 4*Sp3p - 20*zeta3 + zeta2*(-32*S1 + 32*Sp1p + 32*(-(1/n) + np.log(2))) # pylint: disable=line-too-long
    gqq1p_cfnf = (-12 + n*(20 + n*(47 + 3*n*(2 + n))))/(9.*np.power(n,2)*np.power(1 + n,2)) - (40*S1)/9. + (8*S2)/3. # pylint: disable=line-too-long
    # fmt: on
    result = CF * CA * gqq1p_cfca + CF * CF * gqq1p_cfcf + CF * nf * gqq1p_cfnf
    return result
