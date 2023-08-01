# pylint: skip-file
# fmt: off
r"""The unpolarized, space-like anomalous dimension :math:`\gamma_{gg}^{(3)}`."""
import numba as nb
import numpy as np

from .....harmonics import cache as c
from .....harmonics.log_functions import lm11, lm11m1, lm11m2


@nb.njit(cache=True)
def gamma_gg_nf3(n, cache):
    r"""Return the part proportional to :math:`nf^3` of :math:`\gamma_{gg}^{(3)}`.

    The expression is copied exact from :eqref:`3.14` of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^3}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3=  c.get(c.S3, cache, n)
    S21 = c.get(c.S21, cache, n)
    return 3.0 * (
        -0.0205761316872428
        + 2.599239604033225 / (-1.0 + n)
        - 1.1851851851851851 / np.power(n, 4)
        - 3.753086419753086 / np.power(n, 3)
        - 5.679012345679013 / np.power(n, 2)
        - 2.8050009209056537 / n
        - 1.1851851851851851 / np.power(1.0 + n, 4)
        + 3.753086419753086 / np.power(1.0 + n, 3)
        - 5.679012345679013 / np.power(1.0 + n, 2)
        + 2.8050009209056537 / (1.0 + n)
        - 2.599239604033225 / (2.0 + n)
        + 2.454258338353606 * S1
        - (8.674897119341564 * S1) / (-1.0 + n)
        + (2.3703703703703702 * S1) / np.power(n, 3)
        + (7.506172839506172 * S1) / np.power(n, 2)
        + (7.901234567901234 * S1) / n
        - (2.3703703703703702 * S1) / np.power(1.0 + n, 3)
        + (7.506172839506172 * S1) / np.power(1.0 + n, 2)
        - (7.901234567901234 * S1) / (1.0 + n)
        + (8.674897119341564 * S1) / (2.0 + n)
        - (2.567901234567901 * S2) / (-1.0 + n)
        + (2.3703703703703702 * S2) / np.power(n, 2)
        + (1.6296296296296295 * S2) / n
        + (2.3703703703703702 * S2) / np.power(1.0 + n, 2)
        - (1.6296296296296295 * S2) / (1.0 + n)
        + (2.567901234567901 * S2) / (2.0 + n)
        + (2.567901234567901 * (np.power(S1, 2) + S2)) / (-1.0 + n)
        - (2.3703703703703702 * (np.power(S1, 2) + S2)) / np.power(n, 2)
        - (1.6296296296296295 * (np.power(S1, 2) + S2)) / n
        - (2.3703703703703702 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 2)
        + (1.6296296296296295 * (np.power(S1, 2) + S2)) / (1.0 + n)
        - (2.567901234567901 * (np.power(S1, 2) + S2)) / (2.0 + n)
    ) + 1.3333333333333333 * (
        -0.6337448559670782
        - 54.23172286962781 / (-1.0 + n)
        + 3.5555555555555554 / np.power(n, 5)
        + 13.62962962962963 / np.power(n, 4)
        + 27.65432098765432 / np.power(n, 3)
        + 61.00190529209603 / np.power(n, 2)
        + 26.28917081074211 / n
        + 10.666666666666666 / np.power(1.0 + n, 5)
        - 31.40740740740741 / np.power(1.0 + n, 4)
        + 31.604938271604937 / np.power(1.0 + n, 3)
        - 12.479576189385448 / np.power(1.0 + n, 2)
        + 44.82194030036901 / (1.0 + n)
        - 16.879388241483305 / (2.0 + n)
        + (48.724279835390945 * S1) / (-1.0 + n)
        - (7.111111111111111 * S1) / np.power(n, 4)
        - (27.25925925925926 * S1) / np.power(n, 3)
        - (36.34567901234568 * S1) / np.power(n, 2)
        - (56.49382716049383 * S1) / n
        + (7.111111111111111 * S1) / np.power(1.0 + n, 4)
        - (17.77777777777778 * S1) / np.power(1.0 + n, 3)
        + (3.950617283950617 * S1) / np.power(1.0 + n, 2)
        + (4.345679012345679 * S1) / (1.0 + n)
        + (3.4238683127572016 * S1) / (2.0 + n)
        + (27.65432098765432 * S2) / (-1.0 + n)
        - (7.111111111111111 * S2) / np.power(n, 3)
        - (27.25925925925926 * S2) / np.power(n, 2)
        - (16.59259259259259 * S2) / n
        + (7.111111111111111 * S2) / np.power(1.0 + n, 3)
        - (27.25925925925926 * S2) / np.power(1.0 + n, 2)
        + (16.59259259259259 * S2) / (1.0 + n)
        - (27.65432098765432 * S2) / (2.0 + n)
        - (15.012345679012345 * (np.power(S1, 2) + S2)) / (-1.0 + n)
        + (7.111111111111111 * (np.power(S1, 2) + S2)) / np.power(n, 3)
        + (8.296296296296296 * (np.power(S1, 2) + S2)) / np.power(n, 2)
        + (22.51851851851852 * (np.power(S1, 2) + S2)) / n
        + (4.7407407407407405 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 2)
        - (15.407407407407407 * (np.power(S1, 2) + S2)) / (1.0 + n)
        + (7.901234567901234 * (np.power(S1, 2) + S2)) / (2.0 + n)
        - (9.481481481481481 * S21) / (-1.0 + n)
        + (14.222222222222221 * S21) / np.power(n, 2)
        + (21.333333333333332 * S21) / n
        + (14.222222222222221 * S21) / np.power(1.0 + n, 2)
        - (21.333333333333332 * S21) / (1.0 + n)
        - (28.444444444444443 * S21) / (n * (1.0 + n))
        + (9.481481481481481 * S21) / (2.0 + n)
        + (4.7407407407407405 * S3) / (-1.0 + n)
        - (7.111111111111111 * S3) / np.power(n, 2)
        - (10.666666666666666 * S3) / n
        - (7.111111111111111 * S3) / np.power(1.0 + n, 2)
        + (10.666666666666666 * S3) / (1.0 + n)
        + (14.222222222222221 * S3) / (n * (1.0 + n))
        - (4.7407407407407405 * S3) / (2.0 + n)
        - (9.481481481481481 * (S1 * S2 - 1.0 * S21 + S3)) / (-1.0 + n)
        + (14.222222222222221 * (S1 * S2 - 1.0 * S21 + S3)) / np.power(n, 2)
        + (21.333333333333332 * (S1 * S2 - 1.0 * S21 + S3)) / n
        + (14.222222222222221 * (S1 * S2 - 1.0 * S21 + S3)) / np.power(1.0 + n, 2)
        - (21.333333333333332 * (S1 * S2 - 1.0 * S21 + S3)) / (1.0 + n)
        - (28.444444444444443 * (S1 * S2 - 1.0 * S21 + S3)) / (n * (1.0 + n))
        + (9.481481481481481 * (S1 * S2 - 1.0 * S21 + S3)) / (2.0 + n)
        + (1.5802469135802468 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (-1.0 + n)
        - (2.3703703703703702 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / np.power(n, 2)
        - (3.5555555555555554 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3)) / n
        - (2.3703703703703702 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / np.power(1.0 + n, 2)
        + (3.5555555555555554 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (1.0 + n)
        + (4.7407407407407405 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (n * (1.0 + n))
        - (1.5802469135802468 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (2.0 + n)
    )


@nb.njit(cache=True)
def gamma_gg_nf1(n, cache, variation):
    r"""Return the part proportional to :math:`nf^1` of :math:`\gamma_{gg}^{(3)}`.

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^1}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    common = 18143.980574437464 + 1992.766087237516/np.power(-1. + n,3) + 20005.925925925927/np.power(n,7) - 19449.679012345678/np.power(n,6) + 80274.123066115/np.power(n,5) - 11714.245609287387*S1 + 13880.514502193577*lm11(n,S1)
    if variation == 1:
        fit = 51906.450933224565/n - 55794.44458990475/(1. + n) - (3244.182054400047*S1)/np.power(n,2) + (5896.657744251454*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 2:
        fit = 143243.25209661626/n - 140219.52798151976/(2. + n) - (81141.96014226894*S1)/np.power(n,2) + (4359.928069421606*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 3:
        fit = -4846.510890091015/n + 73944.02374603187/np.power(1. + n,3) + (15528.258395929412*S1)/np.power(n,2) + (5875.765469829209*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 4:
        fit = 3769.9068671191385/n - (15787.280088316747*S1)/np.power(n,2) + (3319.747916876306*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 209289.30375366632*lm11m2(n,S1)
    elif variation == 5:
        fit = -4404.650816024677/n + (32166.525125055265*S1)/np.power(n,2) + (5984.665222488821*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 26795.61010300351*lm11m1(n,S1)
    elif variation == 6:
        fit = 18002.20882549217/(-1. + n) - 34510.86180174593/n + (14377.94953322174*S1)/np.power(n,2) + (12777.152485988116*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 7:
        fit = 32712.59210667494/np.power(n,3) - 10863.21459339799/n + (19949.446087845266*S1)/np.power(n,2) + (9104.304296741177*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 8:
        fit = 23133.58893729803/np.power(n,2) - 6436.943948938613/n + (9529.55663925907*S1)/np.power(n,2) + (6913.820202625404*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 9:
        fit = 3314.1531443520435/(-1. + n) + 35997.28048068957/n - 45522.85315535845/(1. + n) + (7163.336378194301*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 10:
        fit = 15292.461183783358/(-1. + n) - 7754.767835203314/n - 21106.273066417427/(2. + n) + (11510.166071508424*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 11:
        fit = 18943.68531934485/(-1. + n) - 34231.50197430196/n + 38795.17200645132/np.power(1. + n,2) + (12574.453319273369*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 12:
        fit = 243015.55817041075/(-1. + n) - 405291.74871837825/n - 924241.7199143948/np.power(1. + n,3) + (99039.0230533651*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 13:
        fit = 9421.639301348665/(-1. + n) - 16264.723220366097/n + (8269.376900788462*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 99755.61548732712*lm11m2(n,S1)
    elif variation == 14:
        fit = 32552.831422747156/(-1. + n) - 58844.770942175055/n + (18267.307590060325*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 21658.053945007472*lm11m1(n,S1)
    elif variation == 15:
        fit = 26670.108040113628/(-1. + n) - 192284.97382369108/np.power(n,4) - 17027.576353111184/n - (2060.8605568773337*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 16:
        fit = 64459.179128132346/(-1. + n) - 84418.96962498086/np.power(n,3) - 95536.59785086475/n + (22255.40045589177*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 17:
        fit = -35383.49147569926/(-1. + n) + 68602.85078732096/np.power(n,2) + 48742.57555689275/n - (4610.608264940457*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    else:
        fit = 23311.07841529563/(-1. + n) - 11310.8808131583/np.power(n,4) - 3041.5516187238777/np.power(n,3) + 5396.261160271705/np.power(n,2) - 24256.141353532745/n - 50017.51153931546/np.power(1. + n,3) + 2282.06894155596/np.power(1. + n,2) - 5959.841043839012/(1. + n) - 9489.753002819834/(2. + n) - (507.1580296279411*S1)/np.power(n,2) + (13331.74331502859*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 302.2091857644728*lm11m1(n,S1) - 18179.112896529026*lm11m2(n,S1)
    return common + fit



@nb.njit(cache=True)
def gamma_gg_nf2(n, cache, variation):
    r"""Return the part proportional to :math:`nf^2` of :math:`\gamma_{gg}^{(3)}`.

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^2}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    common = -423.811346198137 - 568.8888888888889/np.power(n,7) + 1725.6296296296296/np.power(n,6) - 2196.543209876543/np.power(n,5) + 440.0487580115612*S1 - 135.11111111111114*lm11(n,S1)
    if variation == 1:
        fit = -2376.754718471023/n + 1986.9752104021475/(1. + n) - (29.413328132453657*S1)/np.power(n,2) + (243.6914020341996*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 2:
        fit = -5629.479269151885/n + 4993.556762890926/(2. + n) + (2744.7150845388774*S1)/np.power(n,2) + (298.41806508138217*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 3:
        fit = -355.6443895005434/n - 2633.32565133846/np.power(1. + n,3) - (697.9453672577569*S1)/np.power(n,2) + (244.43542649166247*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 4:
        fit = -662.4958800366925/n + (417.27675074971785*S1)/np.power(n,2) + (335.4613978477791*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 7453.298646800853*lm11m2(n,S1)
    elif variation == 5:
        fit = -371.3800961661366/n - (1290.4743148587395*S1)/np.power(n,2) + (240.55724273082328*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 954.2565288285368*lm11m1(n,S1)
    elif variation == 6:
        fit = -641.1022267836108/(-1. + n) + 700.7749426995084/n - (656.9800845794034*S1)/np.power(n,2) - (1.339668489438944*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 7:
        fit = -1164.9745787725433/np.power(n,3) - 141.37500847741444/n - (855.3945615089527*S1)/np.power(n,2) + (129.45934487552083*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 8:
        fit = -823.8430919763521/np.power(n,2) - 299.005218970897/n - (484.3170791242564*S1)/np.power(n,2) + (207.4677850666592*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 9:
        fit = 30.047719973994777/(-1. + n) - 2520.994974716468/n + 2080.1024406993206/(1. + n) + (255.17572494624633*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 10:
        fit = -517.28413779068/(-1. + n) - 521.8069586029795/n + 964.421319764148/(2. + n) + (56.55348517779473*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 11:
        fit = -684.1216646478402/(-1. + n) + 688.0099901838175/n - 1772.6905583568107/np.power(1. + n,2) + (7.922383473211929*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 12:
        fit = -10922.769229617297/(-1. + n) + 17643.083571901912/n + 42231.91922591065/np.power(1. + n,3) - (3942.953936253496*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 13:
        fit = -249.02522869092658/(-1. + n) - 132.95657379723917/n + (204.6367663552972*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 4558.191872124068*lm11m2(n,S1)
    elif variation == 14:
        fit = -1305.9723629973262/(-1. + n) + 1812.678531475261/n - (252.20454712133116*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 989.6341672183413*lm11m1(n,S1)
    elif variation == 15:
        fit = -1037.169631665525/(-1. + n) + 8786.190136094137/np.power(n,4) - 98.09907798298099/n + (676.6623679879976*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 16:
        fit = -2763.8878304060045/(-1. + n) + 3857.4055136430484/np.power(n,3) + 3489.2597080989826/n - (434.43481453536947*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 17:
        fit = 1798.2819022378217/(-1. + n) - 3134.710315160434/np.power(n,2) - 3103.373221197692/n + (793.169486129463*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    else:
        fit = -958.4119229639646/(-1. + n) + 516.8347138878904/np.power(n,4) + 158.3782902865003/np.power(n,3) - 232.85608277275213/np.power(n,2) + 477.6730210169136/n + 2329.329033798364/np.power(1. + n,3) - 104.27591519745945/np.power(1. + n,2) + 239.23986182949812/(1. + n) + 350.46929897971023/(2. + n) - (50.14899412782161*S1)/np.power(n,2) - (55.13659342362342*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 2.0810375523414395*lm11m1(n,S1) + 706.5582658191131*lm11m2(n,S1)
    return common + fit


@nb.njit(cache=True)
def gamma_gg_nf0(n, cache, variation):
    r"""Return the part proportional to :math:`nf^0` of :math:`\gamma_{gg}^{(3)}`.

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^0}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    common = -68587.9129845144 - 49851.703887834694/np.power(-1. + n,4) + 213823.9810748423/np.power(-1. + n,3) - 103680./np.power(n,7) - 17280./np.power(n,6) - 627978.8224813186/np.power(n,5) + 40880.33011934297*S1 - 85814.12027987762*lm11(n,S1)
    if variation == 1:
        fit = 657693.1275908262/n - 1.1706414373839432e6/(1. + n) - (370650.30059459625*S1)/np.power(n,2) + (287643.02359540213*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 2:
        fit = 2.5740603391615152e6/n - 2.941991644365525e6/(2. + n) - (2.0050489863420033e6*S1)/np.power(n,2) + (255400.3971361596*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 3:
        fit = -533059.2724548094/n + 1.5514436765207502e6/np.power(1. + n,3) + (23220.39448094448*S1)/np.power(n,2) + (287204.67597244744*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 4:
        fit = -352275.4015418936/n - (633821.1309449758*S1)/np.power(n,2) + (233576.03813655287*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 4.3911671345781535e6*lm11m2(n,S1)
    elif variation == 5:
        fit = -523788.46214332484/n + (372313.29472429893*S1)/np.power(n,2) + (289489.537196911*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 562207.434039533*lm11m1(n,S1)
    elif variation == 6:
        fit = 377710.2141700686/(-1. + n) - 1.1554566929419546e6/n - (914.6109012422813*S1)/np.power(n,2) + (432004.92760893324*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 7:
        fit = 686353.5630791902/np.power(n,3) - 659297.6949952773/n + (115982.7752636412*S1)/np.power(n,2) + (354943.7019770036*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 8:
        fit = 485373.3737194109/np.power(n,2) - 566428.6671276395/n - (102640.3233464204*S1)/np.power(n,2) + (308984.44084716233*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 9:
        fit = 378644.55156064907/(-1. + n) - 1.1599418606928566e6/n + 2895.802190713511/(1. + n) + (432362.03398212773*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 10:
        fit = 377882.58678340475/(-1. + n) - 1.1571587030349977e6/n + 1342.6133808775278/(2. + n) + (432085.52321715665*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 11:
        fit = 377650.3249167252/(-1. + n) - 1.1554744635946492e6/n - 2467.8405744727033/np.power(1. + n,2) + (432017.82171933743*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 12:
        fit = 363396.65285306936/(-1. + n) - 1.1318705591614237e6/n + 58792.91414672283/np.power(1. + n,3) + (426517.6328449669*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 13:
        fit = 378256.0418217043/(-1. + n) - 1.1566173672939881e6/n + (432291.67647271376*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 6345.659593752382*lm11m2(n,S1)
    elif variation == 14:
        fit = 376784.61918064323/(-1. + n) - 1.1539087630012901e6/n + (431655.6875663606*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 1377.713295886333*lm11m1(n,S1)
    elif variation == 15:
        fit = 377158.8312978629/(-1. + n) + 12231.642128156545/np.power(n,4) - 1.1565688406394941e6/n + (432948.8040951983*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 16:
        fit = 374754.9908692327/(-1. + n) + 5370.064049746925/np.power(n,3) - 1.1515747205286513e6/n + (431401.9967878436*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    elif variation == 17:
        fit = 381106.18821765727/(-1. + n) - 4363.968245037315/np.power(n,2) - 1.1607526150905455e6/n + (433110.99878386786*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n
    else:
        fit = 221373.2353924128/(-1. + n) + 719.5083604797968/np.power(n,4) + 40689.6251252316/np.power(n,3) + 28294.67091025727/np.power(n,2) - 643671.8010288504/n + 94719.79945102782/np.power(1. + n,3) - 145.16709261604137/np.power(1. + n,2) - 68690.9197172488/(1. + n) - 172979.3547638028/(2. + n) - (153032.87574472665*S1)/np.power(n,2) + (372566.99517294974*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 33152.06749031878*lm11m1(n,S1) - 257930.6749990824*lm11m2(n,S1)
    return common + fit


@nb.njit(cache=True)
def gamma_gg(n, nf, cache, variation):
    r"""Compute the |N3LO| gluon-gluon singlet anomalous dimension.

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
        |N3LO| gluon-gluon singlet anomalous dimension
        :math:`\gamma_{gg}^{(3)}(N)`

    """
    return (
        gamma_gg_nf0(n, cache, variation)
        + nf * gamma_gg_nf1(n, cache, variation)
        + nf**2 * gamma_gg_nf2(n, cache, variation)
        + nf**3 * gamma_gg_nf3(n, cache)
    )
