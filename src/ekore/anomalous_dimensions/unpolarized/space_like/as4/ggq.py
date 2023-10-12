# pylint: skip-file
# fmt: off
r"""The unpolarized, space-like anomalous dimension :math:`\gamma_{gq}^{(3)}`."""
import numba as nb
import numpy as np

from .....harmonics import cache as c
from .....harmonics.log_functions import (
    lm11,
    lm11m1,
    lm11m2,
    lm12,
    lm12m1,
    lm12m2,
    lm13,
    lm13m1,
    lm13m2,
    lm14,
    lm14m1,
    lm14m2,
    lm15,
)


@nb.njit(cache=True)
def gamma_gq_nf3(n, cache):
    r"""Return the part proportional to :math:`nf^3` of :math:`\gamma_{gq}^{(3)}`.

    The expression is copied exact from :eqref:`3.13` of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^3}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    return 1.3333333333333333 * (
        -11.39728026699467 / (-1.0 + n)
        + 11.39728026699467 / n
        - 2.3703703703703702 / np.power(1.0 + n, 4)
        + 6.320987654320987 / np.power(1.0 + n, 3)
        - 3.1604938271604937 / np.power(1.0 + n, 2)
        - 5.698640133497335 / (1.0 + n)
        - (6.320987654320987 * S1) / (-1.0 + n)
        + (6.320987654320987 * S1) / n
        - (2.3703703703703702 * S1) / np.power(1.0 + n, 3)
        + (6.320987654320987 * S1) / np.power(1.0 + n, 2)
        - (3.1604938271604937 * S1) / (1.0 + n)
        + (6.320987654320987 * (np.power(S1, 2) + S2)) / (-1.0 + n)
        - (6.320987654320987 * (np.power(S1, 2) + S2)) / n
        - (1.1851851851851851 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 2)
        + (3.1604938271604937 * (np.power(S1, 2) + S2)) / (1.0 + n)
        - (0.7901234567901234 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (-1.0 + n)
        + (0.7901234567901234 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3)) / n
        - (0.3950617283950617 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (1.0 + n)
    )


@nb.njit(cache=True)
def gamma_gq_nf0(n, cache, variation):
    r"""Return the part proportional to :math:`nf^0` of :math:`\gamma_{gq}^{(3)}`.

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^0}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    S5 = c.get(c.S5, cache, n)
    S2m2 = ((-1 + 2 * n - 2 * n**2)/((-1 + n)**2 * n**2) + S2)/n
    S1m2 = ((1 - 2 * n)/((-1 + n) * n) + S1)/n
    common = -22156.31283903764/np.power(-1. + n,4) + 95032.88047770769/np.power(-1. + n,3) - 37609.87654320987/np.power(n,7) - 35065.67901234568/np.power(n,6) - 175454.58483973087/np.power(n,5) - 375.3983146907502*lm14(n,S1,S2,S3,S4) - 13.443072702331962*lm15(n,S1,S2,S3,S4,S5)
    if variation == 1:
        fit = 54395.612515252/(-1. + n) - 2.1646239683351885e6/np.power(n,4) + (2855.303350475541*np.power(S1,3))/n - 60918.73535555526*S2m2
    elif variation == 2:
        fit = -226090.67195519924/(-1. + n) + 1.1897625270895162e6/np.power(n,3) + (767.5822028088761*np.power(S1,3))/n + 166822.85829328437*S2m2
    elif variation == 3:
        fit = -156969.41163838003/(-1. + n) + 308065.6508918336/np.power(n,2) + (2298.0372229048253*np.power(S1,3))/n + 79897.51027543494*S2m2
    elif variation == 4:
        fit = 166341.6709608748/(-1. + n) - 491341.2274352182/n + (1928.2261659049982*np.power(S1,3))/n + 192393.6054838923*S2m2
    elif variation == 5:
        fit = -2910.286549470067/(-1. + n) - 231374.96742185086/(1. + n) + (2346.7938301139043*np.power(S1,3))/n + 124908.19081096651*S2m2
    elif variation == 6:
        fit = -31627.458190721398/(-1. + n) - 195967.18171506765/(2. + n) + (2692.8548253146964*np.power(S1,3))/n + 109472.35691430997*S2m2
    elif variation == 7:
        fit = -162008.96605774807/(-1. + n) + 720879.4649423409/np.power(1. + n,2) + (3458.480965363866*np.power(S1,3))/n + 46219.18000891417*S2m2
    elif variation == 8:
        fit = -127936.1111804959/(-1. + n) + 1.317743273735378e6/np.power(1. + n,3) + (1810.811048345219*np.power(S1,3))/n + 80332.05372142064*S2m2
    elif variation == 9:
        fit = -31133.856975192168/(-1. + n) + (4363.35894123817*np.power(S1,3))/n + 108577.87074947657*S2m2 + 69739.16311988933*lm11(n,S1)
    elif variation == 10:
        fit = -43609.29566283978/(-1. + n) + (7720.209697473076*np.power(S1,3))/n + 101242.36850860748*S2m2 - 25996.353887666173*lm12(n,S1,S2)
    elif variation == 11:
        fit = -36336.05360209993/(-1. + n) + (16069.407610813529*np.power(S1,3))/n + 106190.63289422497*S2m2 + 11885.534727629803*lm13(n,S1,S2,S3)
    elif variation == 12:
        fit = -56096.82600570976/(-1. + n) - (30904.014914347616*np.power(S1,2))/n + (8763.722478566568*np.power(S1,3))/n + 94447.37306501447*S2m2
    elif variation == 13:
        fit = -1.744831439637831e6/np.power(n,4) + 230734.49573806854/np.power(n,3) + (2450.4248345695696*np.power(S1,3))/n - 16752.071359429487*S2m2
    elif variation == 14:
        fit = -1.607549555980178e6/np.power(n,4) + 79281.89558453669/np.power(n,2) + (2711.8887412487193*np.power(S1,3))/n - 24679.125874904974*S2m2
    elif variation == 15:
        fit = -3.2164345301161744e6/np.power(n,4) + 238747.19120481954/n + (3305.7786240482187*np.power(S1,3))/n - 184005.5159797561*S2m2
    elif variation == 16:
        fit = -109930.67245290388/np.power(n,4) - 219624.56359674424/(1. + n) + (2372.618543477443*np.power(S1,3))/n + 115470.94939990941*S2m2
    elif variation == 17:
        fit = -795851.0838465183/np.power(n,4) - 123917.39558697924/(2. + n) + (2752.5810580044063*np.power(S1,3))/n + 46825.94005648564*S2m2
    elif variation == 18:
        fit = -1.6205225107818602e6/np.power(n,4) + 181200.78744997882/np.power(1. + n,2) + (3006.918515153242*np.power(S1,3))/n - 33988.47090234153*S2m2
    elif variation == 19:
        fit = -1.5188447027408783e6/np.power(n,4) + 393126.6104427561/np.power(1. + n,3) + (2543.6965856722713*np.power(S1,3))/n - 18778.930858748437*S2m2
    elif variation == 20:
        fit = -787951.7251390232/np.power(n,4) + (3814.406780618335*np.power(S1,3))/n + 46878.866608639015*S2m2 + 44353.186297165696*lm11(n,S1)
    elif variation == 21:
        fit = -963193.8684382912/np.power(n,4) + (5555.469755339871*np.power(S1,3))/n + 29085.456626461673*S2m2 - 14428.742592292887*lm12(n,S1,S2)
    elif variation == 22:
        fit = -866884.6931575488/np.power(n,4) + (10777.447027986002*np.power(S1,3))/n + 39266.975234472426*S2m2 + 7125.637269182096*lm13(n,S1,S2,S3)
    elif variation == 23:
        fit = -1.0989759638298883e6/np.power(n,4) - (15214.098294404646*np.power(S1,2))/n + (5764.027981513135*np.power(S1,3))/n + 15568.25999702346*S2m2
    elif variation == 24:
        fit = -1.30410210442889e6/np.power(n,4) + (3218.5602518731034*np.power(S1,3))/n - 20261.445878695344*S1m2 + 8032.736655052195*S2m2
    else:
        fit = -27249.23559757206/(-1. + n) - 741654.0341202153/np.power(n,4) + 59187.37595114936/np.power(n,3) + 16139.481103182095/np.power(n,2) - 10524.751509599944/n + 71286.24517408892/np.power(1. + n,3) + 37586.67718301332/np.power(1. + n,2) - 18791.647125774794/(1. + n) - 13328.524054251953/(2. + n) - (1921.5880503646777*np.power(S1,2))/n + (4306.191959951149*np.power(S1,3))/n - 844.2269116123059*S1m2 + 48854.5972905356*S2m2 + 4753.847892377293*lm11(n,S1) - 1684.379019998294*lm12(n,S1,S2) + 792.1321665338291*lm13(n,S1,S2,S3)
    return common + fit


@nb.njit(cache=True)
def gamma_gq_nf1(n, cache, variation):
    r"""Return the part proportional to :math:`nf^1` of :math:`\gamma_{gq}^{(3)}`.

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^1}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    S5 = c.get(c.S5, cache, n)
    S3m2 = (-(((-1 + 2 * n) * (1 - n + n**2))/((-1 + n)**3 * n**3)) + S3)/n
    S2m2 = ((-1 + 2 * n - 2 * n**2)/((-1 + n)**2 * n**2) + S2)/n
    S1m2 = ((1 - 2 * n)/((-1 + n) * n) + S1)/n
    common = 5309.62962962963/np.power(n,7) + 221.23456790123456/np.power(n,6) + 9092.91243376357/np.power(n,5) + 34.49474165523548*lm14(n,S1,S2,S3,S4) + 0.5486968449931413*lm15(n,S1,S2,S3,S4,S5)
    if variation == 1:
        fit = 61414.641491730144/np.power(n,4) - (275.8821081210985*np.power(S1,3))/n - 1723.0775384248207*S2m2 + 5359.279437479722*S3m2
    elif variation == 2:
        fit = 30628.764498540346/np.power(n,3) - (270.06312683778754*np.power(S1,3))/n + 1773.2287929788677*S2m2 + 579.1073675469371*S3m2
    elif variation == 3:
        fit = 14964.683461563509/np.power(n,2) - (218.2558990293935*np.power(S1,3))/n + 1752.1522405673672*S2m2 - 1437.7646853596411*S3m2
    elif variation == 4:
        fit = 7580.473213968261/n - (247.331894047618*np.power(S1,3))/n - 6197.2716578690515*S2m2 + 4215.915142027546*S3m2
    elif variation == 5:
        fit = 11506.23542429956/(1. + n) - (274.1023835976748*np.power(S1,3))/n - 10030.195812335383*S2m2 + 7245.875500079163*S3m2
    elif variation == 6:
        fit = 15648.586212999671/(2. + n) - (319.57482516808324*np.power(S1,3))/n - 13077.961868048167*S2m2 + 9906.744899797963*S3m2
    elif variation == 7:
        fit = 32890.76649300865/np.power(1. + n,2) - (166.9129330699367*np.power(S1,3))/n - 70.8880877568522*S2m2 - 1177.148738100283*S3m2
    elif variation == 8:
        fit = 102016.12308266855/np.power(1. + n,3) - (240.30275153122795*np.power(S1,3))/n + 4585.802226607708*S2m2 - 3985.368690549865*S3m2
    elif variation == 9:
        fit = (-450.823181671801*np.power(S1,3))/n - 12903.050719269673*S2m2 + 9834.061598926022*S3m2 - 5511.504206466589*lm11(n,S1)
    elif variation == 10:
        fit = (-882.1406501674703*np.power(S1,3))/n - 15619.249097143265*S2m2 + 12292.391588412715*S3m2 + 2777.986504535076*lm12(n,S1,S2)
    elif variation == 11:
        fit = (-1513.783497714283*np.power(S1,3))/n - 13902.310762445584*S2m2 + 10684.561917950059*S3m2 - 1053.755322341623*lm13(n,S1,S2,S3)
    elif variation == 12:
        fit = (5100.183003924889*np.power(S1,2))/n - (1401.3877591921307*np.power(S1,3))/n - 21387.15682013914*S2m2 + 17430.91515626977*S3m2
    elif variation == 13:
        fit = 31147.78863209503/np.power(n,4) + 15094.71171421834/np.power(n,3) - (273.0143514100448*np.power(S1,3))/n + 3003.4769582959475*S3m2
    elif variation == 14:
        fit = 30964.22640709071/np.power(n,4) + 7419.742457931761/np.power(n,2) - (247.31006171693753*np.power(S1,3))/n + 1989.190320913944*S3m2
    elif variation == 15:
        fit = 85066.31740469787/np.power(n,4) - 2919.3510109124736/n - (286.87721277546535*np.power(S1,3))/n + 5799.605736056626*S3m2
    elif variation == 16:
        fit = 74153.37781346358/np.power(n,4) - 2386.6442197774195/(1. + n) - (276.25126180951514*np.power(S1,3))/n + 4967.958260585389*S3m2
    elif variation == 17:
        fit = 70734.17185530512/np.power(n,4) - 2374.6369076942033/(2. + n) - (269.2518389654746*np.power(S1,3))/n + 4669.21202970862*S3m2
    elif variation == 18:
        fit = -2635.0286245103766/np.power(n,4) + 34301.96273361259/np.power(1. + n,2) - (162.23755129655925*np.power(S1,3))/n - 1457.5977471044935*S3m2
    elif variation == 19:
        fit = 44641.11065487181/np.power(n,4) + 27862.583657909938/np.power(1. + n,3) - (266.1646951326012*np.power(S1,3))/n + 2807.074625599327*S3m2
    elif variation == 20:
        fit = 70879.97629826797/np.power(n,4) - (248.919875787699*np.power(S1,3))/n + 4669.617977026223*S3m2 + 849.4429232949681*lm11(n,S1)
    elif variation == 21:
        fit = 69029.84607075108/np.power(n,4) - (200.7081314711859*np.power(S1,3))/n + 4499.597346756129*S3m2 - 344.4607838774497*lm12(n,S1,S2)
    elif variation == 22:
        fit = 70103.38136051684/np.power(n,4) - (100.74792356621695*np.power(S1,3))/n + 4605.876127181845*S3m2 + 149.08180946409476*lm13(n,S1,S2,S3)
    elif variation == 23:
        fit = 66796.13877765737/np.power(n,4) - (446.9068014839647*np.power(S1,2))/n - (177.25895475872284*np.power(S1,3))/n + 4301.494627873232*S3m2
    elif variation == 24:
        fit = 60675.73486178171/np.power(n,4) - (248.51519590078777*np.power(S1,3))/n - 682.8975276953039*S1m2 + 4145.5785035058925*S3m2
    else:
        fit = 30540.48679182162/np.power(n,4) + 1905.1448421982786/np.power(n,3) + 932.6844133123028/np.power(n,2) + 194.21342512732446/n + 5411.6127808574365/np.power(1. + n,3) + 2799.6970511092186/np.power(1. + n,2) + 379.9829668550892/(1. + n) + 553.0812210543945/(2. + n) + (193.88650843503848*np.power(S1,2))/n - (375.74241936415484*np.power(S1,3))/n - 28.454063653970994*S1m2 - 3616.665795969917*S2m2 + 4789.568969203283*S3m2 - 194.2525534654842*lm11(n,S1) + 101.3969050274011*lm12(n,S1,S2) - 37.694729703230344*lm13(n,S1,S2,S3)
    return common + fit



@nb.njit(cache=True)
def gamma_gq_nf2(n, cache, variation):
    r"""Return the part proportional to :math:`nf^2` of :math:`\gamma_{gq}^{(3)}`.

    This therm is parametrized using the analytic result from :cite:`Falcioni:2023tzp`
    with an higher number of moments (30).

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^2}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    Lm11 = lm11(n,S1)
    Lm12 = lm12(n,S1, S2)
    Lm13 = lm13(n,S1, S2, S3)
    Lm14 = lm14(n,S1, S2, S3, S4)
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, S2)
    Lm13m1 = lm13m1(n, S1, S2, S3)
    Lm14m1 = lm14m1(n, S1, S2, S3, S4)
    Lm11m2 = lm11m2(n, S1)
    Lm12m2 = lm12m2(n, S1, S2)
    Lm13m2 = lm13m2(n, S1, S2, S3)
    Lm14m2 = lm14m2(n, S1, S2, S3, S4)
    return (
        -(70.60121231446594/(-1. + n)**2)
        - 699.5449657900476/(-1. + n)
        + 617.4606265472538/n**5
        + 21.0418422974213/n**4
        + 656.9409510996688/n**3
        + 440.98264702900605/n**2
        - 485.09325526270226/n
        + 468.97972206118425/(1 + n)**2
        + 131.12265149192916/(1 + n)
        - 284.0960143480868/(2 + n)**2
        + 189.98763175661884/(2 + n)
        + 355.07676818390956/(3 + n)**2
        + 259.2485292950681/(3 + n)
        + 592.4002328363352/(n + n**2)
        + 54.543536161068644/(3 + 4 * n + n**2)
        - 62.424886245567585/(6 + 5 * n + n**2)
        + 154.10095015747495 * (1/n - n/(2 + 3 * n + n**2))
        - 645.1788277783346 * Lm11
        + 32.22330776302828 * Lm11m1
        - 476.25599212133864 * Lm11m2
        - 212.9330738830414 * Lm12
        - 93.58928584449357 * Lm12m1
        + 105.52933047599603 * Lm12m2
        - 26.13260173754001 * Lm13
        - 22.482518440225107 * Lm13m1
        - 45.725204763960996 * Lm13m2
        - 0.877914951989026 * Lm14
        - 0.40377681107870367 * Lm14m1
        + 20.629383319025006 * Lm14m2
    )

@nb.njit(cache=True)
def gamma_gq(n, nf, cache, variation):
    r"""Compute the |N3LO| gluon-quark singlet anomalous dimension.

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
        |N3LO| gluon-quark singlet anomalous dimension
        :math:`\gamma_{gq}^{(3)}(N)`

    """
    return (
        gamma_gq_nf0(n, cache, variation)
        + nf * gamma_gq_nf1(n, cache, variation)
        + nf**2 * gamma_gq_nf2(n, cache, variation)
        + nf**3 * gamma_gq_nf3(n, cache)
    )
