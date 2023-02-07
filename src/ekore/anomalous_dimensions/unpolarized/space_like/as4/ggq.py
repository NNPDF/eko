# pylint: skip-file
# fmt: off
r"""The anomalous dimension :math:`\gamma_{gq}^{(3)}`."""
import numba as nb
import numpy as np

from .....harmonics.log_functions import lm11, lm12, lm13, lm14, lm15


@nb.njit(cache=True)
def gamma_gq_nf3(n, sx):
    r"""Implement the part proportional to :math:`nf^3` of :math:`\gamma_{gq}^{(3)}`.

    The expression is copied exact from Eq. 3.13 of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^3}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
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
def gamma_gq_nf0(n, sx, variation):
    r"""Implement the part proportional to :math:`nf^0` of :math:`\gamma_{gq}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^0}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    S5 = sx[4][0]
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
def gamma_gq_nf1(n, sx, variation):
    r"""Implement the part proportional to :math:`nf^1` of :math:`\gamma_{gq}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    S5 = sx[4][0]
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
def gamma_gq_nf2(n, sx, variation):
    r"""Implement the part proportional to :math:`nf^2` of :math:`\gamma_{gq}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^2}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    S2m2 = ((-1 + 2 * n - 2 * n**2)/((-1 + n)**2 * n**2) + S2)/n
    S1m2 = ((1 - 2 * n)/((-1 + n) * n) + S1)/n
    common = 778.5349794238683/np.power(n,5) - 0.877914951989026*lm14(n,S1,S2,S3,S4)
    if variation == 1:
        fit = -186.22900377040247/(-1. + n) - 478.90721340293356/np.power(n,4) + (5.74075848054696*np.power(S1,3))/n + 170.87183353958662*S2m2
    elif variation == 2:
        fit = -248.28454175688805/(-1. + n) + 263.22625305579476/np.power(n,3) + (5.278865453482085*np.power(S1,3))/n + 221.25799463997834*S2m2
    elif variation == 3:
        fit = -232.99196870519415/(-1. + n) + 68.15727099578979/np.power(n,2) + (5.617467427605377*np.power(S1,3))/n + 202.02639746547175*S2m2
    elif variation == 4:
        fit = -161.46175750229568/(-1. + n) - 108.70565119071854/n + (5.535649437328786*np.power(S1,3))/n + 226.91533534247935*S2m2
    elif variation == 5:
        fit = (-147.71749614319918 - 250.09753306427467*n)/(-1. + np.power(n,2)) + (5.6282544701748725*np.power(S1,3))/n + 211.98468116923237*S2m2
    elif variation == 6:
        fit = -205.26097869205503/(-1. + n) - 43.35630496864558/(2. + n) + (5.7048179330688145*np.power(S1,3))/n + 208.5696158709912*S2m2
    elif variation == 7:
        fit = -234.10693324364027/(-1. + n) + 159.4893065978856/np.power(1. + n,2) + (5.874207114101223*np.power(S1,3))/n + 194.5753133344904*S2m2
    elif variation == 8:
        fit = -226.56856342946034/(-1. + n) + 291.54105675473517/np.power(1. + n,3) + (5.509672201751474*np.power(S1,3))/n + 202.1225370231326*S2m2
    elif variation == 9:
        fit = -205.1517730350374/(-1. + n) + (6.074404744837193*np.power(S1,3))/n + 208.37171735240508*S2m2 + 15.429279525384134*lm11(n,S1)
    elif variation == 10:
        fit = -207.91187255591723/(-1. + n) + (6.817083410967055*np.power(S1,3))/n + 206.74879115899574*S2m2 - 5.751503069858567*lm12(n,S1,S2)
    elif variation == 11:
        fit = -206.30272097177453/(-1. + n) + (8.664282354210393*np.power(S1,3))/n + 207.84355844549347*S2m2 + 2.629587586332136*lm13(n,S1,S2,S3)
    elif variation == 12:
        fit = -210.6746472670261/(-1. + n) - (6.837287160307751*np.power(S1,2))/n + (7.047952977901978*np.power(S1,3))/n + 205.24544813934457*S2m2
    elif variation == 13:
        fit = -1916.1103405421652/np.power(n,4) - 789.943403334499/np.power(n,3) + (7.126901942835579*np.power(S1,3))/n + 19.662692663483778*S2m2
    elif variation == 14:
        fit = -2386.1090637317598/np.power(n,4) - 271.4297670165366/np.power(n,2) + (6.231753223826018*np.power(S1,3))/n + 46.80178342814188*S2m2
    elif variation == 15:
        fit = 3122.074998548171/np.power(n,4) - 817.375694750788/n + (4.198509803834483*np.power(S1,3))/n + 592.272153767969*S2m2
    elif variation == 16:
        fit = -7513.36216027104/np.power(n,4) + 751.9074019187506/(1. + n) + (7.3932798253001035*np.power(S1,3))/n - 433.0164903242173*S2m2
    elif variation == 17:
        fit = -5165.042709988898/np.power(n,4) + 424.2440164363194/(2. + n) + (6.092438869963565*np.power(S1,3))/n - 198.00320317266343*S2m2
    elif variation == 18:
        fit = -2341.694811608125/np.power(n,4) - 620.3596313904335/np.power(1. + n,2) + (5.221688287772126*np.power(S1,3))/n + 78.67328868471006*S2m2
    elif variation == 19:
        fit = -2689.7993005817502/np.power(n,4) - 1345.9095988275376/np.power(1. + n,3) + (6.807576458948666*np.power(S1,3))/n + 26.60185572989274*S2m2
    elif variation == 20:
        fit = -5192.086981291262/np.power(n,4) + (2.457168719366107*np.power(S1,3))/n - 198.18440269638717*S2m2 - 151.8477192594809*lm11(n,S1)
    elif variation == 21:
        fit = -4592.127384254676/np.power(n,4) - (3.5035403946824317*np.power(S1,3))/n - 137.2668238433194*S2m2 + 49.39829214844644*lm12(n,S1,S2)
    elif variation == 22:
        fit = -4921.851803874727/np.power(n,4) - (21.38151952256833*np.power(S1,3))/n - 172.1243055327945*S2m2 - 24.395355958105238*lm13(n,S1,S2,S3)
    elif variation == 23:
        fit = -4127.263341266998/np.power(n,4) + (52.087038597781934*np.power(S1,2))/n - (4.217561046210457*np.power(S1,3))/n - 90.9893035495142*S2m2
    elif variation == 24:
        fit = -3424.992789569337/np.power(n,4) + (4.497110919064199*np.power(S1,3))/n + 69.36715493145202*S1m2 - 65.19066024035324*S2m2
    else:
        fit = -96.87269837207046/(-1. + n) - 1734.4697042431458/np.power(n,4) - 21.946547928279337/np.power(n,3) - 8.469687334197783/np.power(n,2) - 38.586722747562774/n - 43.93202258636677/np.power(1. + n,3) - 19.202930199689494/np.power(1. + n,2) + 31.32947507994794/(1. + n) + 15.870321311153074/(2. + n) + (-6.1548956726333 - 10.420730544344778*n)/(-1. + np.power(n,2)) + (1.8854063098947575*np.power(S1,2))/n + (3.934050962226076*np.power(S1,3))/n + 2.890298122143834*S1m2 + 80.65707534985623*S2m2 - 5.684101655587364*lm11(n,S1) + 1.818616211607828*lm12(n,S1,S2) - 0.9069070154905459*lm13(n,S1,S2,S3)
    return common + fit


@nb.njit(cache=True)
def gamma_gq(n, nf, sx, variation):
    r"""Compute the |N3LO| gluon-quark singlet anomalous dimension.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx : list
        harmonic sums cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| gluon-quark singlet anomalous dimension
        :math:`\gamma_{gq}^{(3)}(N)`

    See Also
    --------
    gamma_gq_nf0: :math:`\gamma_{gq}^{(3)}|_{nf^0}`
    gamma_gq_nf1: :math:`\gamma_{gq}^{(3)}|_{nf^1}`
    gamma_gq_nf2: :math:`\gamma_{gq}^{(3)}|_{nf^2}`
    gamma_gq_nf3: :math:`\gamma_{gq}^{(3)}|_{nf^3}`

    """
    return (
        gamma_gq_nf0(n, sx, variation)
        + nf * gamma_gq_nf1(n, sx, variation)
        + nf**2 * gamma_gq_nf2(n, sx, variation)
        + nf**3 * gamma_gq_nf3(n, sx)
    )
