"""Compute the |NLO| Altarelli-Parisi splitting kernels.

These expression have been obtained using the procedure described in the
`wiki <https://github.com/N3PDF/eko/wiki/Parse-NLO-expressions>`_
involving ``FormGet`` :cite:`Hahn:2016ebn`.
"""

import numba as nb
import numpy as np

from eko import constants

from .... import harmonics
from ....harmonics.constants import log2, zeta2, zeta3


@nb.njit(cache=True)
def gamma_nsm(n, nf, sx):
    r"""Compute the |NLO| valence-like non-singlet anomalous dimension.

    Implements Eq. (3.5) of :cite:`Moch:2004pa`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx : numpy.ndarray
        List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
    gamma_nsm : complex
        |NLO| valence-like non-singlet anomalous dimension
        :math:`\\gamma_{ns,-}^{(1)}(N)`
    """
    S1 = sx[0]
    S2 = sx[1]
    # Here, Sp refers to S' ("s-prime") (german: "s-strich" or in Pegasus language: SSTR)
    # of :cite:`Gluck:1989ze` and NOT to the Spence function a.k.a. dilogarithm
    # TODO : these harmonic sums are computed also for the QED sector then we can use
    # the ones that are passed to the O(as1aem1) anomalous dimensions
    Sp1m = harmonics.S1((n - 1) / 2)
    Sp2m = harmonics.S2((n - 1) / 2)
    Sp3m = harmonics.S3((n - 1) / 2)
    g3n = harmonics.g_functions.mellin_g3(n, S1)
    # fmt: off
    gqq1m_cfca = 16*g3n - (144 + n*(1 + n)*(156 + n*(340 + n*(655 + 51*n*(2 + n)))))/(18.*np.power(n,3)*np.power(1 + n,3)) + (-14.666666666666666 + 8/n - 8/(1 + n))*S2 - (4*Sp2m)/(n + np.power(n,2)) + S1*(29.77777777777778 + 16/np.power(n,2) - 16*S2 + 8*Sp2m) + 2*Sp3m + 10*zeta3 + zeta2*(16*S1 - 16*Sp1m - (16*(1 + n*log2))/n) # pylint: disable=line-too-long
    gqq1m_cfcf = -32*g3n + (24 - n*(-32 + 3*n*(-8 + n*(3 + n)*(3 + np.power(n,2)))))/(2.*np.power(n,3)*np.power(1 + n,3)) + (12 - 8/n + 8/(1 + n))*S2 + S1*(-24/np.power(n,2) - 8/np.power(1 + n,2) + 16*S2 - 16*Sp2m) + (8*Sp2m)/(n + np.power(n,2)) - 4*Sp3m - 20*zeta3 + zeta2*(-32*S1 + 32*Sp1m + 32*(1/n + log2)) # pylint: disable=line-too-long
    gqq1m_cfnf = (-12 + n*(20 + n*(47 + 3*n*(2 + n))))/(9.*np.power(n,2)*np.power(1 + n,2)) - (40*S1)/9. + (8*S2)/3. # pylint: disable=line-too-long
    # fmt: on
    result = constants.CF * (
        (constants.CA * gqq1m_cfca)
        + (constants.CF * gqq1m_cfcf)
        + (2.0 * constants.TR * nf * gqq1m_cfnf)
    )
    return result


@nb.njit(cache=True)
def gamma_nsp(n, nf, sx):
    r"""Compute the |NLO| singlet-like non-singlet anomalous dimension.

    Implements Eq. (3.5) of :cite:`Moch:2004pa`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx : numpy.ndarray
        List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
    gamma_nsp : complex
        |NLO| singlet-like non-singlet anomalous dimension
        :math:`\\gamma_{ns,+}^{(1)}(N)`
    """
    S1 = sx[0]
    S2 = sx[1]
    Sp1p = harmonics.S1(n / 2)
    Sp2p = harmonics.S2(n / 2)
    Sp3p = harmonics.S3(n / 2)
    g3n = harmonics.g_functions.mellin_g3(n, S1)
    # fmt: off
    gqq1p_cfca = -16*g3n + (132 - n*(340 + n*(655 + 51*n*(2 + n))))/(18.*np.power(n,2)*np.power(1 + n,2)) + (-14.666666666666666 + 8/n - 8/(1 + n))*S2 - (4*Sp2p)/(n + np.power(n,2)) + S1*(29.77777777777778 - 16/np.power(n,2) - 16*S2 + 8*Sp2p) + 2*Sp3p + 10*zeta3 + zeta2*(16*S1 - 16*Sp1p + 16*(1/n - log2)) # pylint: disable=line-too-long
    gqq1p_cfcf = 32*g3n - (8 + n*(32 + n*(40 + 3*n*(3 + n)*(3 + np.power(n,2)))))/(2.*np.power(n,3)*np.power(1 + n,3)) + (12 - 8/n + 8/(1 + n))*S2 + S1*(40/np.power(n,2) - 8/np.power(1 + n,2) + 16*S2 - 16*Sp2p) + (8*Sp2p)/(n + np.power(n,2)) - 4*Sp3p - 20*zeta3 + zeta2*(-32*S1 + 32*Sp1p + 32*(-(1/n) + log2)) # pylint: disable=line-too-long
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
    r"""Compute the |NLO| pure-singlet quark-quark anomalous dimension.

    Implements Eq. (3.6) of :cite:`Vogt:2004mw`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors

    Returns
    -------
    gamma_ps : complex
        |NLO| pure-singlet quark-quark anomalous dimension
        :math:`\\gamma_{ps}^{(1)}(N)`
    """
    # fmt: off
    gqqps1_nfcf = (-4*(2 + n*(5 + n))*(4 + n*(4 + n*(7 + 5*n))))/((-1 + n)*np.power(n,3)*np.power(1 + n,3)*np.power(2 + n,2)) # pylint: disable=line-too-long
    # fmt: on
    result = 2.0 * constants.TR * nf * constants.CF * gqqps1_nfcf
    return result


@nb.njit(cache=True)
def gamma_qg(n, nf, sx):
    r"""Compute the |NLO| quark-gluon singlet anomalous dimension.

    Implements Eq. (3.7) of :cite:`Vogt:2004mw`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx : numpy.ndarray
        List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
    gamma_qg : complex
        |NLO| quark-gluon singlet anomalous dimension
        :math:`\\gamma_{qg}^{(1)}(N)`
    """
    S1 = sx[0]
    S2 = sx[1]
    Sp2p = harmonics.S2(n / 2)
    # fmt: off
    gqg1_nfca = (-4*(16 + n*(64 + n*(104 + n*(128 + n*(85 + n*(36 + n*(25 + n*(15 + n*(6 + n))))))))))/((-1 + n)*np.power(n,3)*np.power(1 + n,3)*np.power(2 + n,3)) - (16*(3 + 2*n)*S1)/np.power(2 + 3*n + np.power(n,2),2) + (4*(2 + n + np.power(n,2))*np.power(S1,2))/(n*(2 + 3*n + np.power(n,2))) - (4*(2 + n + np.power(n,2))*S2)/(n*(2 + 3*n + np.power(n,2))) + (4*(2 + n + np.power(n,2))*Sp2p)/(n*(2 + 3*n + np.power(n,2))) # pylint: disable=line-too-long
    gqg1_nfcf = (-2*(4 + n*(8 + n*(1 + n)*(25 + n*(26 + 5*n*(2 + n))))))/(np.power(n,3)*np.power(1 + n,3)*(2 + n)) + (8*S1)/np.power(n,2) - (4*(2 + n + np.power(n,2))*np.power(S1,2))/(n*(2 + 3*n + np.power(n,2))) + (4*(2 + n + np.power(n,2))*S2)/(n*(2 + 3*n + np.power(n,2))) # pylint: disable=line-too-long
    # fmt: on
    result = (
        2.0 * constants.TR * nf * (constants.CA * gqg1_nfca + constants.CF * gqg1_nfcf)
    )
    return result


@nb.njit(cache=True)
def gamma_gq(n, nf, sx):
    r"""Compute the |NLO| gluon-quark singlet anomalous dimension.

    Implements Eq. (3.8) of :cite:`Vogt:2004mw`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx : numpy.ndarray
        List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
    gamma_gq : complex
        |NLO| gluon-quark singlet anomalous dimension
        :math:`\\gamma_{gq}^{(1)}(N)`
    """
    S1 = sx[0]
    S2 = sx[1]
    Sp2p = harmonics.S2(n / 2)
    # fmt: off
    ggq1_cfcf = (-8 + 2*n*(-12 + n*(-1 + n*(28 + n*(43 + 6*n*(5 + 2*n))))))/((-1 + n)*np.power(n,3)*np.power(1 + n,3)) - (4*(10 + n*(17 + n*(8 + 5*n)))*S1)/((-1 + n)*n*np.power(1 + n,2)) + (4*(2 + n + np.power(n,2))*np.power(S1,2))/(n*(-1 + np.power(n,2))) + (4*(2 + n + np.power(n,2))*S2)/(n*(-1 + np.power(n,2))) # pylint: disable=line-too-long
    ggq1_cfca = (-4*(144 + n*(432 + n*(-152 + n*(-1304 + n*(-1031 + n*(695 + n*(1678 + n*(1400 + n*(621 + 109*n))))))))))/(9.*np.power(n,3)*np.power(1 + n,3)*np.power(-2 + n + np.power(n,2),2)) + (4*(-12 + n*(-22 + 41*n + 17*np.power(n,3)))*S1)/(3.*np.power(-1 + n,2)*np.power(n,2)*(1 + n)) + ((8 + 4*n + 4*np.power(n,2))*np.power(S1,2))/(n - np.power(n,3)) + ((8 + 4*n + 4*np.power(n,2))*S2)/(n - np.power(n,3)) + (4*(2 + n + np.power(n,2))*Sp2p)/(n*(-1 + np.power(n,2))) # pylint: disable=line-too-long
    ggq1_cfnf = (8*(16 + n*(27 + n*(13 + 8*n))))/(9.*(-1 + n)*n*np.power(1 + n,2)) - (8*(2 + n + np.power(n,2))*S1)/(3.*n*(-1 + np.power(n,2))) # pylint: disable=line-too-long
    # fmt: on
    result = constants.CF * (
        (constants.CA * ggq1_cfca)
        + (constants.CF * ggq1_cfcf)
        + (2.0 * constants.TR * nf * ggq1_cfnf)
    )
    return result


@nb.njit(cache=True)
def gamma_gg(n, nf, sx):
    r"""Compute the |NLO| gluon-gluon singlet anomalous dimension.

    Implements Eq. (3.9) of :cite:`Vogt:2004mw`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx : numpy.ndarray
        List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
    gamma_gg : complex
        |NLO| gluon-gluon singlet anomalous dimension
        :math:`\\gamma_{gg}^{(1)}(N)`
    """
    S1 = sx[0]
    Sp1p = harmonics.S1(n / 2)
    Sp2p = harmonics.S2(n / 2)
    Sp3p = harmonics.S3(n / 2)
    g3n = harmonics.g_functions.mellin_g3(n, S1)
    # fmt: off
    ggg1_caca = 16*g3n - (2*(576 + n*(1488 + n*(560 + n*(-1248 + n*(-1384 + n*(1663 + n*(4514 + n*(4744 + n*(3030 + n*(1225 + 48*n*(7 + n))))))))))))/(9.*np.power(-1 + n,2)*np.power(n,3)*np.power(1 + n,3)*np.power(2 + n,3)) + S1*(29.77777777777778 + 16/np.power(-1 + n,2) + 16/np.power(1 + n,2) - 16/np.power(2 + n,2) - 8*Sp2p) + (16*(1 + n + np.power(n,2))*Sp2p)/(n*(1 + n)*(-2 + n + np.power(n,2))) - 2*Sp3p - 10*zeta3 + zeta2*(-16*S1 + 16*Sp1p + 16*(-(1/n) + log2)) # pylint: disable=line-too-long
    ggg1_canf = (8*(6 + n*(1 + n)*(28 + n*(1 + n)*(13 + 3*n*(1 + n)))))/(9.*np.power(n,2)*np.power(1 + n,2)*(-2 + n + np.power(n,2))) - (40*S1)/9. # pylint: disable=line-too-long
    ggg1_cfnf = (2*(-8 + n*(-8 + n*(-10 + n*(-22 + n*(-3 + n*(6 + n*(8 + n*(4 + n)))))))))/(np.power(n,3)*np.power(1 + n,3)*(-2 + n + np.power(n,2))) # pylint: disable=line-too-long
    # fmt: on
    result = constants.CA * constants.CA * ggg1_caca + 2.0 * constants.TR * nf * (
        constants.CA * ggg1_canf + constants.CF * ggg1_cfnf
    )
    return result


@nb.njit(cache=True)
def gamma_singlet(n, nf, sx):
    r"""Compute the next-leading-order singlet anomalous dimension matrix.

    .. math::
        \\gamma_S^{(1)} = \\left(\begin{array}{cc}
        \\gamma_{qq}^{(1)} & \\gamma_{qg}^{(1)}\\
        \\gamma_{gq}^{(1)} & \\gamma_{gg}^{(1)}
        \\end{array}\right)

    Parameters
    ----------
    N : complex
        Mellin moment
    sx : numpy.ndarray
        List of harmonic sums: :math:`S_{1},S_{2}`
    nf : int
        Number of active flavors

    Returns
    -------
    gamma_S_1 : numpy.ndarray
        |NLO| singlet anomalous dimension matrix :math:`\\gamma_{S}^{(1)}(N)`

    See Also
    --------
    gamma_nsp : :math:`\\gamma_{qq}^{(1)}`
    gamma_ps : :math:`\\gamma_{qq}^{(1)}`
    gamma_qg : :math:`\\gamma_{qg}^{(1)}`
    gamma_gq : :math:`\\gamma_{gq}^{(1)}`
    gamma_gg : :math:`\\gamma_{gg}^{(1)}`
    """
    gamma_qq = gamma_nsp(n, nf, sx) + gamma_ps(n, nf)
    gamma_S_0 = np.array(
        [[gamma_qq, gamma_qg(n, nf, sx)], [gamma_gq(n, nf, sx), gamma_gg(n, nf, sx)]],
        np.complex_,
    )
    return gamma_S_0


@nb.njit(cache=True)
def gamma_singlet_qed(N, nf, sx):
    r"""Compute the leading-order singlet anomalous dimension matrix for the unified evolution basis.

    .. math::
        \\gamma_S^{(2,0)} = \\left(\begin{array}{cccc}
        \\gamma_{gg}^{(2,0)} & 0 & \\gamma_{gq}^{(2,0)} & 0\\
        0 & 0 & 0 & 0 \\
        \\gamma_{qg}^{(2,0)} & 0 & \\gamma_{qq}^{(2,0)} & 0 \\
        0 & 0 & 0 & \\gamma_{qq}^{(2,0)} \\
        \\end{array}\right)

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        Number of active flavors
    s1 : complex
        harmonic sum :math:`S_{1}`

    Returns
    -------
    gamma_S : numpy.ndarray
        Leading-order singlet anomalous dimension matrix :math:`\\gamma_{S}^{(2,0)}(N)`

    See Also
    --------
    gamma_ns : :math:`\\gamma_{qq}^{(0)}`
    gamma_qg : :math:`\\gamma_{qg}^{(0)}`
    gamma_gq : :math:`\\gamma_{gq}^{(0)}`
    gamma_gg : :math:`\\gamma_{gg}^{(0)}`
    """
    gamma_ns_p = gamma_nsp(N, nf, sx)
    gamma_qq = gamma_ns_p + gamma_ps(N, nf)
    gamma_S = np.array(
        [
            [gamma_gg(N, nf, sx), 0.0 + 0.0j, gamma_gq(N, nf, sx), 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [gamma_qg(N, nf, sx), 0.0 + 0.0j, gamma_qq, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, gamma_ns_p],
        ],
        np.complex_,
    )
    return gamma_S


@nb.njit(cache=True)
def gamma_valence_qed(N, nf, sx):
    r"""Compute the leading-order valence anomalous dimension matrix for the unified evolution basis.

    .. math::
        \\gamma_V^{(2,0)} = \\left(\begin{array}{cc}
        \\gamma_{ns-}^{(2,0)} & 0\\
        0 & \\gamma_{ns-}^{(2,0)}
        \\end{array}\right)

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        Number of active flavors
    s1 : complex
        harmonic sum :math:`S_{1}`

    Returns
    -------
    gamma_V : numpy.ndarray
        Leading-order singlet anomalous dimension matrix :math:`\\gamma_{V}^{(2,0)}(N)`

    See Also
    --------
    gamma_ns : :math:`\\gamma_{qq}^{(0)}`
    gamma_qg : :math:`\\gamma_{qg}^{(0)}`
    gamma_gq : :math:`\\gamma_{gq}^{(0)}`
    gamma_gg : :math:`\\gamma_{gg}^{(0)}`
    """
    gamma_V = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        np.complex_,
    )
    return gamma_V * gamma_nsm(N, nf, sx)
