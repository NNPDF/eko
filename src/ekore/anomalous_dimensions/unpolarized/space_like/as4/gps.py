# pylint: skip-file
# fmt: off
r"""The unpolarized, space-like anomalous dimension
:math:`\gamma_{ps}^{(3)}`."""
import numba as nb
import numpy as np

from .....harmonics import cache as c
from .....harmonics.log_functions import (
    lm11m1,
    lm11m2,
    lm12m1,
    lm12m2,
    lm13m1,
    lm13m2,
    lm14m1,
    lm14m2,
)


@nb.njit(cache=True)
def gamma_ps_nf3(n, cache):
    r"""Return the part proportional to :math:`nf^3` of
    :math:`\gamma_{ps}^{(3)}`.

    The expression is copied exact from :eqref:`3.10` of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ps}^{(3)}|_{nf^3}`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    return 1.3333333333333333 * (
        16.305796943701882 / (-1.0 + n)
        + 3.5555555555555554 / np.power(n, 5)
        - 17.185185185185187 / np.power(n, 4)
        + 28.839506172839506 / np.power(n, 3)
        - 48.95252257604665 / np.power(n, 2)
        + 23.09354523864061 / n
        + 39.111111111111114 / np.power(1.0 + n, 5)
        - 61.03703703703704 / np.power(1.0 + n, 4)
        - 10.666666666666666 / np.power(1.0 + n, 3)
        + 59.29439100420026 / np.power(1.0 + n, 2)
        - 94.20465634975173 / (1.0 + n)
        + 18.962962962962962 / np.power(2.0 + n, 4)
        - 34.76543209876543 / np.power(2.0 + n, 3)
        + 14.222222222222221 / np.power(2.0 + n, 2)
        + 54.805314167409236 / (2.0 + n)
        - (1.5802469135802468 * S1) / (-1.0 + n)
        + (7.111111111111111 * S1) / np.power(n, 4)
        - (34.370370370370374 * S1) / np.power(n, 3)
        + (38.71604938271605 * S1) / np.power(n, 2)
        - (37.135802469135804 * S1) / n
        + (35.55555555555556 * S1) / np.power(1.0 + n, 4)
        - (43.851851851851855 * S1) / np.power(1.0 + n, 3)
        - (39.50617283950617 * S1) / np.power(1.0 + n, 2)
        + (89.28395061728395 * S1) / (1.0 + n)
        + (18.962962962962962 * S1) / np.power(2.0 + n, 3)
        - (34.76543209876543 * S1) / np.power(2.0 + n, 2)
        - (50.5679012345679 * S1) / (2.0 + n)
        + (4.7407407407407405 * (np.power(S1, 2) + S2)) / (-1.0 + n)
        + (7.111111111111111 * (np.power(S1, 2) + S2)) / np.power(n, 3)
        - (15.407407407407407 * (np.power(S1, 2) + S2)) / np.power(n, 2)
        + (13.037037037037036 * (np.power(S1, 2) + S2)) / n
        + (14.222222222222221 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 3)
        - (4.7407407407407405 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 2)
        - (20.14814814814815 * (np.power(S1, 2) + S2)) / (1.0 + n)
        + (9.481481481481481 * (np.power(S1, 2) + S2)) / np.power(2.0 + n, 2)
        + (2.3703703703703702 * (np.power(S1, 2) + S2)) / (2.0 + n)
        - (1.5802469135802468 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (-1.0 + n)
        + (2.3703703703703702 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / np.power(n, 2)
        - (1.1851851851851851 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3)) / n
        + (2.3703703703703702 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / np.power(1.0 + n, 2)
        + (1.1851851851851851 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (1.0 + n)
        + (1.5802469135802468 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (2.0 + n)
    )


@nb.njit(cache=True)
def gamma_ps_nf1(n, cache, variation):
    r"""Return the part proportional to :math:`nf^1` of
    :math:`\gamma_{ps}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ps}^{(3)}|_{nf^1}`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    common = -3498.454512979491/np.power(-1. + n,3) + 5404.444444444444/np.power(n,7) + 3425.9753086419755/np.power(n,6) + 20515.223982421852/np.power(n,5) + 247.55054124312667*lm13m1(n,S1,S2,S3) + 199.11111*lm13m2(n,S1,S2,S3) + 56.46090534979424*lm14m1(n,S1,S2,S3,S4) + 13.168724000000001*lm14m2(n,S1,S2,S3,S4)
    if variation == 1:
        fit = -22671.047415335794*(-1./np.power(-1. + n,2) + 1/n**2) - 95766.48344777289*(1/(-1. + n) - 1./n) + 83244.80226112883/np.power(n,4) - 7297.905863348574/np.power(n,3) - 11811.283838661806/(6. + 5.*n + np.power(n,2)) - 125068.51271516114*(-1./np.power(n,2) + np.power(1. + n,-2)) + 37009.89475114885*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) + 4472.524325379381*lm11m1(n,S1) + 1379.5436804403923*lm12m1(n,S1,S2) - 747.0351123958919*lm12m2(n,S1,S2)
    elif variation == 2:
        fit = -22305.851784770348*(-1./np.power(-1. + n,2) + 1/n**2) - 93143.88222127911*(1/(-1. + n) - 1./n) + 76112.81780987934/np.power(n,4) - 947.9931300103644/np.power(n,3) - 22421.029779221768/(3. + 4.*n + np.power(n,2)) - 115368.95599436517*(-1./np.power(n,2) + np.power(1. + n,-2)) + 39532.927714115904*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) + 4344.344850330874*lm11m1(n,S1) + 1369.0296723356619*lm12m1(n,S1,S2) - 537.4842267704546*lm12m2(n,S1,S2)
    elif variation == 3:
        fit = -15491.216025024809*(-1./np.power(-1. + n,2) + 1/n**2) - 44205.53589847306*(1/(-1. + n) - 1./n) - 56971.58553396101/np.power(n,4) + 117542.69212339079/np.power(n,3) + 110200.44311522982/(n + np.power(n,2)) + 65626.83867112151*(-1./np.power(n,2) + np.power(1. + n,-2)) - 23587.264147857193*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) + 1952.5096264407057*lm11m1(n,S1) + 1172.8378019695606*lm12m1(n,S1,S2) + 3372.747225178992*lm12m2(n,S1,S2)
    elif variation == 4:
        fit = -28028.046200402754*(-1./np.power(-1. + n,2) + 1/n**2) - 134237.0158414345*(1/(-1. + n) - 1./n) + 187862.72816484852/np.power(n,4) - 100443.6896697422/np.power(n,3) + 328890.04192783125/(3. + 4.*n + np.power(n,2)) - 185068.90911929717/(6. + 5.*n + np.power(n,2)) - 267349.6912816729*(-1./np.power(n,2) + np.power(1. + n,-2)) + 6352.753831873534*lm11m1(n,S1) + 1533.7707570391663*lm12m1(n,S1,S2) - 3820.8806206608274*lm12m2(n,S1,S2)
    elif variation == 5:
        fit = -18285.943879217793*(-1./np.power(-1. + n,2) + 1/n**2) - 64275.48033036274*(1/(-1. + n) - 1./n) - 2392.774990991716/np.power(n,4) + 68948.86027471385/np.power(n,3) + 67305.24670870126/(n + np.power(n,2)) - 4597.507368550904/(6. + 5.*n + np.power(n,2)) - 8600.759166330034*(-1./np.power(n,2) + np.power(1. + n,-2)) + 2933.4178271943297*lm11m1(n,S1) + 1253.297447894429*lm12m1(n,S1,S2) + 1769.1342276217056*lm12m2(n,S1,S2)
    elif variation == 6:
        fit = -18037.762566201312*(-1./np.power(-1. + n,2) + 1/n**2) - 62493.20197829131*(1/(-1. + n) - 1./n) - 7239.544199486837/np.power(n,4) + 73264.1311363246/np.power(n,3) + 69019.83601756362/(n + np.power(n,2)) - 8378.457901949132/(3. + 4.*n + np.power(n,2)) - 2009.127518527169*(-1./np.power(n,2) + np.power(1. + n,-2)) + 2846.3135759542292*lm11m1(n,S1) + 1246.1526689127147*lm12m1(n,S1,S2) + 1911.5346949746186*lm12m2(n,S1,S2)
    else:
        fit = 20803.31131182547/np.power(-1. + n,2) - 82353.59995293559/(-1. + n) + 46769.407251902856/np.power(n,4) + 25177.68247855468/np.power(n,3) + 54658.39002233034/np.power(n,2) + 91179.52633917019/n - 75461.7013341558/np.power(1. + n,2) + 41087.58764024912/(n + np.power(n,2)) - (8825.926386234592*n)/(2. + 3.*n + np.power(n,2)) + 49681.759041110054/(3. + 4.*n + np.power(n,2)) - 33579.616721084974/(6. + 5.*n + np.power(n,2)) + 3816.977339528842*lm11m1(n,S1) + 1325.7720047653206*lm12m1(n,S1,S2) + 324.66936465802365*lm12m2(n,S1,S2)
    return common + fit


@nb.njit(cache=True)
def gamma_ps_nf2(n, cache):
    r"""Return the part proportional to :math:`nf^2` of
    :math:`\gamma_{ps}^{(3)}`.

    This therm is parametrized using the analytic result from :cite:`Gehrmann:2023cqm`
    with an higher number of moments (30).

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache


    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ps}^{(3)}|_{nf^2}`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, S2)
    Lm13m1 = lm13m1(n, S1, S2, S3)
    Lm14m1 = lm14m1(n, S1, S2, S3, S4)
    Lm11m2 = lm11m2(n, S1)
    Lm12m2 = lm12m2(n, S1, S2)
    Lm13m2 = lm13m2(n, S1, S2, S3)
    Lm14m2 = lm14m2(n, S1, S2, S3, S4)
    return (
        114.43829930315756/(-1 + n)**2
        - 482.02078387833865/(n + n**2)
        - 189.286751823932/(3 + 4 * n + n**2)
        + 240.56786018072623/(6 + 5 * n + n**2)
        - 693.3770662139755 * (1 / n**7 - 1 / (1 + n)**7)
        - 378.3908520452082 * (-(1 / n**6) + 1 / (1 + n)**6)
        - 914.6834189565818 * (1 / n**5 - 1 / (1 + n)**5)
        + 13.600197369492934 * (-(6 / n**4) + 6 / (1 + n)**4)
        - 69.77767807874679 * (2 / n**3 - 2 / (1 + n)**3)
        + 684.9299795769539 * (-(1 / n**2) + 1 / (1 + n)**2)
        - 160.76186945134114 * (1 / n - n / (2 + 3 * n + n**2))
        + 23.29094040522371 * Lm11m2
        - 2.0809106854377792 * Lm12m2
        - 6.14078307251059 * Lm13m2
        - 1.7777777777777777 * (
            -216.38233518950935 * Lm11m1
            - 75.17763559409342 * Lm12m1
            - 13.185185185185185 * Lm13m1
            - 1.6296296296296295 * Lm14m1
        ) - 4 * (
            260.9049778489959 * Lm11m1
               + 87.34510160874684 * Lm12m1
               + 16 * Lm13m1
               + 1.6296296296296295 * Lm14m1
        )
        + 0.8477304323488296 * Lm14m2
        + 356.6141626634052 * (1 / (n - 1) - 1 / n)
    )



@nb.njit(cache=True)
def gamma_ps(n, nf, cache, variation):
    r"""Compute the |N3LO| pure singlet quark-quark anomalous dimension.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| pure singlet quark-quark anomalous dimension
        :math:`\gamma_{ps}^{(3)}(N)`
    """
    return (
        +nf * gamma_ps_nf1(n, cache, variation)
        + nf**2 * gamma_ps_nf2(n, cache)
        + nf**3 * gamma_ps_nf3(n, cache)
    )
