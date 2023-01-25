# -- coding: utf-8 --
"""This file contains the next-leading-order Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from eko import constants
from .... import harmonics
from ....harmonics.constants import log2, zeta2, zeta3

from scipy.special import digamma


@nb.njit(cache=True)
def gamma_nsm(n, nf, sx):
    """
     Computes the |NLO| polarized valence-like non-singlet anomalous dimension.

    :Floratos:1981hs (eq B.18)

     Parameters
     ----------
         n : complex
             Mellin moment
         nf : int
             number of active flavors
         sx : numpy.ndarray
             List of harmonic sums: :math:`S_{1},S_{2}`

     Returns
     -------
         gamma_nsm : complex
             |nLO| valence-like non-singlet anomalous dimension
             :math:`\\gamma_{ns,-}^{(1)}(n)`
    """
    S1 = sx[0]
    S2 = sx[1]
    # Here, Sp refers to S' ("s-prime") (german: "s-strich" or in Pegasus language: SSTR)
    # of :cite:`Gluck:1989ze` and NOT to the Spence function a.k.a. dilogarithm
    Sp1p = harmonics.S1(n / 2)
    Sp2p = harmonics.S2(n / 2)
    Sp3p = harmonics.S3(n / 2)
    Sp1m = harmonics.S1((n - 1) / 2)
    Sp2m = harmonics.S2((n - 1) / 2)
    Sp3m = harmonics.S3((n - 1) / 2)
    g3n = harmonics.g_functions.mellin_g3(n, S1)
    # fmt: off
    gqq1m_cfca = 16*g3n - (144 + n*(1 + n)*(156 + n*(340 + n*(655 + 51*n*(2 + n)))))/(18.*np.power(n,3)*np.power(1 + n,3)) + (-14.666666666666666 + 8/n - 8/(1 + n))*S2 - (4*Sp2m)/(n + np.power(n,2)) + S1*(29.77777777777778 + 16/np.power(n,2) - 16*S2 + 8*Sp2m) + 2*Sp3m + 10*zeta3 + zeta2*(16*S1 - 16*Sp1m - (16*(1 + n*log2))/n) # pylint: disable=line-too-long
    gqq1m_cfcf = 32*g3n - (8 + n*(32 + n*(40 + 3*n*(3 + n)*(3 + np.power(n,2)))))/(2.*np.power(n,3)*np.power(1 + n,3)) + (12 - 8/n + 8/(1 + n))*S2 + S1*(40/np.power(n,2) - 8/np.power(1 + n,2) + 16*S2 - 16*Sp2p) + (8*Sp2p)/(n + np.power(n,2)) - 4*Sp3p - 20*zeta3 + zeta2*(-32*S1 + 32*Sp1p + 32*(-(1/n) + log2))
    gqq1m_cfnf = (-12 + n*(20 + n*(47 + 3*n*(2 + n))))/(9.*np.power(n,2)*np.power(1 + n,2)) - (40*S1)/9. + (8*S2)/3. # pylint: disable=line-too-long
    # fmt: on
    result = constants.CF * (
        (constants.CA * gqq1m_cfca)
        + (constants.CF * gqq1m_cfcf)
        + (2.0 * constants.TR * nf * gqq1m_cfnf)
    )
    return result


def gamma_nsp(n, nf, sx):
    """
    Computes the |NLO| polarized singlet-like non-singlet anomalous dimension.

    :Floratos:1981hs (eq B.18)

    Parameters
    ----------
        n : complex
            Mellin moment
        nf : int
            number of active flavors
        sx : numpy.ndarray
            List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
        gamma_nsp : complex
            |nLO| singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,+}^{(1)}(n)`
    """
    S1 = sx[0]
    S2 = sx[1]
    Sp1p = harmonics.S1(n / 2)
    Sp2p = harmonics.S2(n / 2)
    Sp3p = harmonics.S3(n / 2)
    Sp1m = harmonics.S1((n - 1) / 2)
    Sp2m = harmonics.S2((n - 1) / 2)
    Sp3m = harmonics.S3((n - 1) / 2)
    g3n = harmonics.g_functions.mellin_g3(n, S1)
    # fmt: off
    gqq1p_cfca = -16*g3n + (132 - n*(340 + n*(655 + 51*n*(2 + n))))/(18.*np.power(n,2)*np.power(1 + n,2)) + (-14.666666666666666 + 8/n - 8/(1 + n))*S2 - (4*Sp2p)/(n + np.power(n,2)) + S1*(29.77777777777778 - 16/np.power(n,2) - 16*S2 + 8*Sp2p) + 2*Sp3p + 10*zeta3 + zeta2*(16*S1 - 16*Sp1p + 16*(1/n - log2)) # pylint: disable=line-too-long
    gqq1p_cfcf = -32*g3n + (24 - n*(-32 + 3*n*(-8 + n*(3 + n)*(3 + np.power(n,2)))))/(2.*np.power(n,3)*np.power(1 + n,3)) + (12 - 8/n + 8/(1 + n))*S2 + S1*(-24/np.power(n,2) - 8/np.power(1 + n,2) + 16*S2 - 16*Sp2m) + (8*Sp2m)/(n + np.power(n,2)) - 4*Sp3m - 20*zeta3 + zeta2*(-32*S1 + 32*Sp1m + 32*(1/n + log2)) # pylint: disable=line-too-long
    gqq1p_cfnf = (-12 + n*(20 + n*(47 + 3*n*(2 + n))))/(9.*np.power(n,2)*np.power(1 + n,2)) - (40*S1)/9. + (8*S2)/3. # pylint: disable=line-too-long
    # fmt: on
    result = constants.CF * (
        (constants.CA * gqq1p_cfca)
        + (constants.CF * gqq1p_cfcf)
        + (2.0 * constants.TR * nf * gqq1p_cfnf)
    )
    return result


@nb.njit(cache=True)
def gamma_ps(n, nf):
    """
    Computes the |NLO| polarized pure-singlet quark-quark anomalous dimension.

    :cite:`Gluck:1995yr` (eq A.3)

    Parameters
    ----------
        n : complex
            Mellin moment
        nf : int
            number of active flavors

    Returns
    -------
        gamma_ps : complex
            |nLO| pure-singlet quark-quark anomalous dimension
            :math:`\\gamma_{ps}^{(1)}(n)`
    """
    # fmt: off
    gqqps1_nfcf = (2* (n+2)*(1 + 2*n + np.power(n , 3)))/ (np.power(1 + n,3)*np.power(n,3)) # pylint: disable=line-too-long
    # fmt: on
    result = 4.0 * constants.TR * nf * constants.CF * gqqps1_nfcf
    return result


@nb.njit(cache=True)
def gamma_qg(n, nf, sx):
    """
    Computes the |NLO| polarized quark-gluon singlet anomalous dimension.

    :cite:`Gluck:1995yr` (eq A.4)

    Parameters
    ----------
        n : complex
            Mellin moment
        nf : int
            number of active flavors
        sx : numpy.ndarray
            List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
        gamma_qg : complex
            |nLO| quark-gluon singlet anomalous dimension
            :math:`\\gamma_{qg}^{(1)}(n)`
    """
    S1 = sx[0]
    S2 = sx[1]
    Sp2m = harmonics.S2((n - 1) / 2)
    # fmt: off
    gqg1_nfca =((np.power(S1,2)-S2+Sp2m)*(n-1)/(n*(n+1))-4*S1/(n*np.power(1+n,2))- (-2-7*n+3*np.power(n,2)-4*np.power(n,3)+np.power(n,4)+np.power(n,5))/(np.power(n,3)*np.power(1+n,3)) )*(2.0) # pylint: disable=line-too-long
    gqg1_nfcf = ((-np.power(S1,2)+S2+2*S1/n)*(n-1)/(n*(n+1))-(n-1)*(1+3.5*n+4*np.power(n,2)+5*np.power(n,3)+2.5*np.power(n,4))/(np.power(n,3)*np.power(1+n, 3))+4*(n-1)/(np.power(n,2)*np.power(1+n,2))*2) # pylint: disable=line-too-long
    # fmt: on
    result = (
        4.0 * constants.TR * nf * (constants.CA * gqg1_nfca + constants.CF * gqg1_nfcf)
    )
    return result


@nb.njit(cache=True)
def gamma_gq(n, nf, sx):
    """
    Computes the |NLO| polarized gluon-quark singlet anomalous dimension.

    :cite:`Gluck:1995yr` (eq A.5)

    Parameters
    ----------
        n : complex
            Mellin moment
        nf : int
            number of active flavors
        sx : numpy.ndarray
            List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
        gamma_gq : complex
            |nLO| gluon-quark singlet anomalous dimension
            :math:`\\gamma_{gq}^{(1)}(n)`
    """
    S1 = sx[0]
    S2 = sx[1]
    Sp2m = harmonics.S2((n - 1) / 2)
    # fmt: off
    ggq1_cfcf = ((np.power(S1,2)+S2)*(n+2))/(n*(n+1))-(2* S1 * (n+2) * (1 + 3* n)) / (n * np.power(1+n,2))-((n+2)* (2+ 15*n+ 8*np.power(n,2)-12.* np.power(n,3) - 9.* np.power(n,4)))/(np.power(n,3) * np.power(1+n,3)) + 8* (n+2)/(np.power(n,2)* np.power(1+n, 2))# pylint: disable=line-too-long
    ggq1_cfca =(-np.power(S1,2)-S2 + Sp2m)*(n+2)/(n*(n+1))+ S1*(12+ 22* n+ 11* np.power(n,2))/(3* np.power(n,2)*(n+1))-(36+ 72* n+41* np.power(n,2)+ 254* np.power(n,3)+271* np.power(n,4)+76 *np.power(n,5))/(9* np.power(n,3)* np.power(1+n,3)) # pylint: disable=line-too-long
    ggq1_cfnf = 4*((- S1 * (n+2) )/ (3* n *(n+1)) + ((n+2) * (2+ 5* n))/ (9* n * np.power(1+n, 2) )) # pylint: disable=line-too-long
    # fmt: on
    result = constants.CF * (
        (constants.CA * ggq1_cfca)
        + (constants.CF * ggq1_cfcf)
        + (4.0 * constants.TR * nf * ggq1_cfnf)
    )
    return result


@nb.njit(cache=True)
def gamma_gg(n, nf, sx):
    """
    Computes the |NLO| polarized gluon-gluon singlet anomalous dimension.

    :cite:`Gluck:1995yr` (eq A.6)

    Parameters
    ----------
        n : complex
            Mellin moment
        nf : int
            number of active flavors
        sx : numpy.ndarray
            List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
        gamma_gq : complex
            |nLO| gluon-quark singlet anomalous dimension
            :math:`\\gamma_{gq}^{(1)}(n)`
    """
    S1 = sx[0]
    Sp2m = harmonics.S2((n - 1) / 2)
    Sp3m = harmonics.S3((n - 1) / 2)
    g3n = harmonics.g_functions.mellin_g3(n, S1)
    SSCHLM = ( -5 / 8 * zeta3 + (zeta2 / 2) * (digamma((n + 1) / 2) - digamma(n / 2)) - S1 / (np.power(n, 2))
        - g3n)
    """SSCHLM redefined since I haven't located the expression for the diagamma part and needed to make sure it is correct, I assume approximation is used originally """
    ggg1_caca = (-4 * S1 * Sp2m- Sp3m+ 8 * SSCHLM + 8 * Sp2m / (n * (n + 1))+ 2.0* S1* (72+ 144 * n+ 67 * np.power(n, 2)+ 134 * np.power(n, 3)+ 67 * np.power(n, 4) )/ (9 * np.power(n, 2) * np.power(n + 1, 2)) - (144 + 258 * n+ 7 * np.power(n, 2) + 698 * np.power(n, 3) + 469 * np.power(n, 4)+ 144 * np.power(n, 5) + 48 * np.power(n, 6) )/ (9 * np.power(n, 3) * np.power(1 + n, 3)) ) * (0.5)  # pylint: disable=line-too-long
    ggg1_canf = ( -5 * S1 / 9 + (-3 + 13 * n + 16 * np.power(n, 2) + 6 * np.power(n, 3) + 3 * np.power(n, 4)) / (9 * np.power(n, 2) * np.power(1 + n, 2)) ) * 4  # pylint: disable=line-too-long
    ggg1_cfnf = (4+ 2 * n - 8 * np.power(n, 2)+ np.power(n, 3)+ 5 * np.power(n, 4)+ 3 * np.power(n, 5)+ 
    np.power(n, 6)) / (np.power(n, 3) * np.power(1 + n, 3))  # pylint: disable=line-too-long
    # fmt: on
    result = 4 * (
        constants.CA * constants.CA * ggg1_caca
        + constants.TR * nf * (constants.CA * ggg1_canf + constants.CF * ggg1_cfnf)
    )

    return result


@nb.njit(cache=True)
def gamma_singlet(n, nf, sx):
    gamma_qq = gamma_nsp(n, nf, sx) + gamma_ps(n, nf)
    gamma_S_0 = np.array(
        [[gamma_qq, gamma_qg(n, nf, sx)], [gamma_gq(n, nf, sx), gamma_gg(n, nf, sx)]],
        np.complex_,
    )
    return gamma_S_0
