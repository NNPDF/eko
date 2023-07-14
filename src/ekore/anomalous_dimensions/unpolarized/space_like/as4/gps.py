# pylint: skip-file
# fmt: off
r"""The unpolarized, space-like anomalous dimension :math:`\gamma_{ps}^{(3)}`."""
import numba as nb
import numpy as np

from .....harmonics import cache as c
from .....harmonics.log_functions import (
    lm11m1,
    lm12m1,
    lm12m2,
    lm13m1,
    lm13m2,
    lm14m1,
    lm14m2,
)


@nb.njit(cache=True)
def gamma_ps_nf3(n, cache):
    r"""Return the part proportional to :math:`nf^3` of :math:`\gamma_{ps}^{(3)}`.

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
    r"""Return the part proportional to :math:`nf^1` of :math:`\gamma_{ps}^{(3)}`.

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
def gamma_ps_nf2(n, cache, variation):
    r"""Return the part proportional to :math:`nf^2` of :math:`\gamma_{ps}^{(3)}`.

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ps}^{(3)}|_{nf^2}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    common = -568.8888888888889/np.power(n,7) + 455.1111111111111/np.power(n,6) - 1856.79012345679/np.power(n,5) - 40.559670781893004*lm13m1(n,S1,S2,S3) - 13.695473*lm13m2(n,S1,S2,S3) - 3.6213991769547325*lm14m1(n,S1,S2,S3,S4)
    if variation == 1:
        fit = 25.822615602170636*(-1./np.power(-1. + n,2) + 1/n**2) + 1369.827563834446*(1/(-1. + n) - 1./n) - 968.9773721636304/np.power(n,4) + 18.07807695637113/np.power(n,3) + 108.26356564198636/(6. + 5.*n + np.power(n,2)) + 2244.8862997267374*(-1./np.power(n,2) + np.power(1. + n,-2)) - 707.6478920878866*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) - 720.6014639482447*lm11m1(n,S1) - 220.72213008547274*lm12m1(n,S1,S2) + 78.28234259944801*lm12m2(n,S1,S2)
    elif variation == 2:
        fit = 22.47520493556606*(-1./np.power(-1. + n,2) + 1/n**2) + 1345.7885980600424*(1/(-1. + n) - 1./n) - 903.6049691868469/np.power(n,4) - 40.125877850642645/np.power(n,3) + 205.5137919740281/(3. + 4.*n + np.power(n,2)) + 2155.979224875976*(-1./np.power(n,2) + np.power(1. + n,-2)) - 730.7743407186217*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) - 719.4265378971868*lm11m1(n,S1) - 220.625755952285*lm12m1(n,S1,S2) + 76.361546936374*lm12m2(n,S1,S2)
    elif variation == 3:
        fit = -39.98854097053157*(-1./np.power(-1. + n,2) + 1/n**2) + 897.2140121944313*(1/(-1. + n) - 1./n) + 316.26214619238453/np.power(n,4) - 1126.2252366164616/np.power(n,3) - 1010.1100115754526/(n + np.power(n,2)) + 496.95075914622464*(-1./np.power(n,2) + np.power(1. + n,-2)) - 152.2073406604394*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) - 697.502707179934*lm11m1(n,S1) - 218.82743921117563*lm12m1(n,S1,S2) + 40.51992490089773*lm12m2(n,S1,S2)
    elif variation == 4:
        fit = 128.25116472071446*(-1./np.power(-1. + n,2) + 1/n**2) + 2105.4037585031724*(1/(-1. + n) - 1./n) - 2969.325356129497/np.power(n,4) + 1799.0727530104648/np.power(n,3) - 6288.544180252262/(3. + 4.*n + np.power(n,2)) + 3421.037168319817/(6. + 5.*n + np.power(n,2)) + 4965.374746715422*(-1./np.power(n,2) + np.power(1. + n,-2)) - 756.552384732902*lm11m1(n,S1) - 223.67102825813416*lm12m1(n,S1,S2) + 137.055795153659*lm12m2(n,S1,S2)
    elif variation == 5:
        fit = -58.02277596330755*(-1./np.power(-1. + n,2) + 1/n**2) + 767.7036973043099*(1/(-1. + n) - 1./n) + 668.4563393906011/np.power(n,4) - 1439.7986192735104/np.power(n,3) - 1286.910355933341/(n + np.power(n,2)) - 29.667479340721076/(6. + 5.*n + np.power(n,2)) + 17.96402399364499*(-1./np.power(n,2) + np.power(1. + n,-2)) - 691.1729662094299*lm11m1(n,S1) - 218.30823801383536*lm12m1(n,S1,S2) + 30.17190740021005*lm12m2(n,S1,S2)
    elif variation == 6:
        fit = -56.42130765016638*(-1./np.power(-1. + n,2) + 1/n**2) + 779.204430661265*(1/(-1. + n) - 1./n) + 637.1806800381983/np.power(n,4) - 1411.9523285234507/np.power(n,3) - 1275.8459233345768/(n + np.power(n,2)) - 54.06607559078503/(3. + 4.*n + np.power(n,2)) + 60.49936386423877*(-1./np.power(n,2) + np.power(1. + n,-2)) - 691.7351258214956*lm11m1(n,S1) - 218.3543496213554*lm12m1(n,S1,S2) + 31.090934653960247*lm12m2(n,S1,S2)
    else:
        fit = -3.686060112407607/np.power(-1. + n,2) + 1210.8570100929442/(-1. + n) - 536.6680886431316/np.power(n,4) - 366.82520538287156/np.power(n,3) - 1653.2563429412996/np.power(n,2) - 1475.961939004102/n + 1656.942403053707/np.power(1. + n,2) - 595.4777151405617/(n + np.power(n,2)) + (265.10492891115797*n)/(2. + 3.*n + np.power(n,2)) - 1022.8494106448363/(3. + 4.*n + np.power(n,2)) + 583.2722091035137/(6. + 5.*n + np.power(n,2)) - 712.8318642981988*lm11m1(n,S1) - 220.0848235237097*lm12m1(n,S1,S2) + 65.58040860742483*lm12m2(n,S1,S2)
    return common + fit



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
        + nf**2 * gamma_ps_nf2(n, cache, variation)
        + nf**3 * gamma_ps_nf3(n, cache)
    )
