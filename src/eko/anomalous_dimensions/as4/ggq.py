# pylint: skip-file
# fmt: off
r"""The anomalous dimension :math:`\gamma_{gq}^{(3)}`."""
import numba as nb
import numpy as np

from ...harmonics.log_functions import lm11, lm12, lm13, lm14, lm15


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
    S3m2 = (-(((-1 + 2 * n) * (1 - n + n**2))/((-1 + n)**3 * n**3)) + S3)/n
    S2m2 = ((-1 + 2 * n - 2 * n**2)/((-1 + n)**2 * n**2) + S2)/n
    S1m2 = ((1 - 2 * n)/((-1 + n) * n) + S1)/n
    common = -22156.31283903764/np.power(-1. + n,4) - 37609.87654320987/np.power(n,7) - 35065.67901234568/np.power(n,6) - 175454.58483973087/np.power(n,5) - 375.3983146907502*lm14(n,S1,S2,S3,S4) - 13.443072702331962*lm15(n,S1,S2,S3,S4,S5)
    if variation == 1:
        fit = 15745.624257482406/(-1. + n) + (1903.523033158839*np.power(S1,3))/n + 48813.5368406788*S2m2 - 86016.97391082704*S3m2
    elif variation == 2:
        fit = 255436.93999852816/np.power(n,4) + (1773.6359265362535*np.power(S1,3))/n + 53974.1396409896*S2m2 - 75593.25739766566*S3m2
    elif variation == 3:
        fit = 127391.73736764732/np.power(n,3) + (1797.8383436930474*np.power(S1,3))/n + 68516.04314964218*S2m2 - 95475.03960060292*S3m2
    elif variation == 4:
        fit = 62241.39486319558/np.power(n,2) + (2013.31594674906*np.power(S1,3))/n + 68428.38115369268*S2m2 - 103863.65203974594*S3m2
    elif variation == 5:
        fit = 31528.84775493806/n + (1892.382517611198*np.power(S1,3))/n + 35364.986724512215*S2m2 - 80348.75982900705*S3m2
    elif variation == 6:
        fit = 47856.95228851117/(1. + n) + (1781.0381904795297*np.power(S1,3))/n + 19423.015991113778*S2m2 - 67746.491324651*S3m2
    elif variation == 7:
        fit = 65085.89613911601/(2. + n) + (1591.908350316398*np.power(S1,3))/n + 6746.689691830788*S2m2 - 56679.35292425162*S3m2
    elif variation == 8:
        fit = 136799.89890214667/np.power(1. + n,2) + (2226.862583130518*np.power(S1,3))/n + 60845.957294330874*S2m2 - 102779.69325620876*S3m2
    elif variation == 9:
        fit = 424307.3911659813/np.power(1. + n,3) + (1921.6182602873532*np.power(S1,3))/n + 80214.15184093872*S2m2 - 114459.69486613528*S3m2
    elif variation == 10:
        fit = (1046.0177002885878*np.power(S1,3))/n + 7474.183454399043*S2m2 - 56981.65868575664*S3m2 - 22923.552675601244*lm11(n,S1)
    elif variation == 11:
        fit = (-747.9260832191462*np.power(S1,3))/n - 3823.0802797942483*S2m2 - 46756.92603620328*S3m2 + 11554.25408078299*lm12(n,S1,S2)
    elif variation == 12:
        fit = (-3375.0669771546873*np.power(S1,3))/n + 3318.04215825505*S2m2 - 53444.24165160307*S3m2 - 4382.799093313148*lm13(n,S1,S2,S3)
    elif variation == 13:
        fit = (21212.777740150144*np.power(S1,2))/n - (2907.588492877621*np.power(S1,3))/n - 27813.07131726292*S2m2 - 25384.681496430374*S3m2
    elif variation == 14:
        fit = 54755.869843467/(-1. + n) - 315616.7630493031/np.power(n,3) + (2165.3599488200293*np.power(S1,3))/n + (62584.338958215805*S3m2)/(-1. + n) - (62584.338958215805*n*S3m2)/(-1. + n)
    elif variation == 15:
        fit = 1.2035238985417902e6/np.power(n,4) - 472830.76921110373/np.power(n,3) + (1683.8055516824106*np.power(S1,3))/n - 1799.4732280360502*S3m2
    elif variation == 16:
        fit = -135899.35243826162/np.power(n,3) + 65163.28968624646/n + (1993.2406445021584*np.power(S1,3))/n - 64212.300210536625*S3m2
    elif variation == 17:
        fit = -50400.87962059398/np.power(n,3) + 66790.93137650128/(1. + n) + (1774.3914289452387*np.power(S1,3))/n - 56776.05279709662*S3m2
    elif variation == 18:
        fit = -13914.222396866957/np.power(n,3) + 72194.83155747408/(2. + n) + (1569.4158732905269*np.power(S1,3))/n - 52441.93671808041*S3m2
    elif variation == 19:
        fit = 873527.5448493527/np.power(n,3) - 2.485176389536297e6/np.power(1. + n,3) + (1072.857038574581*np.power(S1,3))/n + 15718.444207770523*S3m2
    elif variation == 20:
        fit = -15598.29959988322/np.power(n,3) + (953.9620938299396*np.power(S1,3))/n - 52268.391557690455*S3m2 - 25730.39439011375*lm11(n,S1)
    elif variation == 21:
        fit = 6732.578663523597/np.power(n,3) - (613.3839251706755*np.power(S1,3))/n - 49331.64978743182*S3m2 + 10943.618523841711*lm12(n,S1,S2)
    elif variation == 22:
        fit = -6483.1919502418/np.power(n,3) - (3638.325323641454*np.power(S1,3))/n - 51305.219635294605*S3m2 - 4605.847529486776*lm13(n,S1,S2,S3)
    elif variation == 23:
        fit = 36781.77149498919/np.power(n,3) + (15088.019889015372*np.power(S1,2))/n - (1548.9922584166047*np.power(S1,3))/n - 45621.84592373968*S3m2
    elif variation == 24:
        fit = 142411.69121426455/np.power(n,3) + (685.2491698695475*np.power(S1,3))/n + 27834.089389192843*S1m2 - 48350.17111706553*S3m2
    else:
        fit = 2937.562254206225/(-1. + n) + 60790.03493917993/np.power(n,4) + 7337.576888480116/np.power(n,3) + 2593.391452633149/np.power(n,2) + 4028.839060049355/n - 85869.54159876314/np.power(1. + n,3) + 5699.995787589444/np.power(1. + n,2) + 4776.995152708852/(1. + n) + 5720.030320691254/(2. + n) + (1512.533234548563*np.power(S1,2))/n + (708.9641475535428*np.power(S1,3))/n + 1159.7537245497017*S1m2 + 17561.79068097194*S2m2 - 57163.2924910954*S3m2 + (2607.6807899256582*S3m2)/(-1. + n) - (2607.6807899256582*n*S3m2)/(-1. + n) - 2027.2477944047914*lm11(n,S1) + 937.4113585260293*lm12(n,S1,S2) - 374.5269426166634*lm13(n,S1,S2,S3)
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
        fit = 3785.716618913236/(-1. + n) - (244.6533825981967*np.power(S1,3))/n - 2963.8400172742627*S2m2 + 2853.107805957123*S3m2
    elif variation == 2:
        fit = 61414.641491730144/np.power(n,4) - (275.8821081210985*np.power(S1,3))/n - 1723.0775384248207*S2m2 + 5359.279437479722*S3m2
    elif variation == 3:
        fit = 30628.76449854033/np.power(n,3) - (270.06312683778776*np.power(S1,3))/n + 1773.2287929788836*S2m2 + 579.1073675469207*S3m2
    elif variation == 4:
        fit = 14964.683461563514/np.power(n,2) - (218.25589902939427*np.power(S1,3))/n + 1752.1522405674307*S2m2 - 1437.7646853597082*S3m2
    elif variation == 5:
        fit = 7580.473213968261/n - (247.331894047618*np.power(S1,3))/n - 6197.2716578690515*S2m2 + 4215.915142027546*S3m2
    elif variation == 6:
        fit = 11506.23542429956/(1. + n) - (274.1023835976748*np.power(S1,3))/n - 10030.195812335383*S2m2 + 7245.875500079163*S3m2
    elif variation == 7:
        fit = 15648.586212999671/(2. + n) - (319.57482516808324*np.power(S1,3))/n - 13077.961868048167*S2m2 + 9906.744899797963*S3m2
    elif variation == 8:
        fit = 32890.76649300866/np.power(1. + n,2) - (166.912933069938*np.power(S1,3))/n - 70.88808775679364*S2m2 - 1177.1487381003358*S3m2
    elif variation == 9:
        fit = 102016.1230826685/np.power(1. + n,3) - (240.3027515312275*np.power(S1,3))/n + 4585.80222660763*S2m2 - 3985.3686905497766*S3m2
    elif variation == 10:
        fit = (-450.8231816718022*np.power(S1,3))/n - 12903.050719269622*S2m2 + 9834.061598925966*S3m2 - 5511.50420646659*lm11(n,S1)
    elif variation == 11:
        fit = (-882.1406501674719*np.power(S1,3))/n - 15619.249097143183*S2m2 + 12292.39158841263*S3m2 + 2777.9865045350784*lm12(n,S1,S2)
    elif variation == 12:
        fit = (-1513.7834977142838*np.power(S1,3))/n - 13902.310762445548*S2m2 + 10684.561917950019*S3m2 - 1053.7553223416235*lm13(n,S1,S2,S3)
    elif variation == 13:
        fit = (5100.183003924888*np.power(S1,2))/n - (1401.3877591921307*np.power(S1,3))/n - 21387.15682013918*S2m2 + 17430.915156269828*S3m2
    elif variation == 14:
        fit = 1417.1087606297594/(-1. + n) + 19163.487282252743/np.power(n,3) - (260.5514874859701*np.power(S1,3))/n - (1430.3348278505869*S3m2)/(-1. + n) + (1430.3348278505869*n*S3m2)/(-1. + n)
    elif variation == 15:
        fit = 31147.78863209503/np.power(n,4) + 15094.71171421834/np.power(n,3) - (273.0143514100448*np.power(S1,3))/n + 3003.4769582959475*S3m2
    elif variation == 16:
        fit = 23814.661991789828/np.power(n,3) + 1686.4578893516236/n - (265.0060194789698*np.power(S1,3))/n + 1388.2023923834984*S3m2
    elif variation == 17:
        fit = 26027.404391768934/np.power(n,3) + 1728.5820543957968/(1. + n) - (270.6699444884554*np.power(S1,3))/n + 1580.656121744779*S3m2
    elif variation == 18:
        fit = 26971.69697164435/np.power(n,3) + 1868.4376408366413/(2. + n) - (275.97481237684514*np.power(S1,3))/n + 1692.8251710670384*S3m2
    elif variation == 19:
        fit = 49939.12505759102/np.power(n,3) - 64317.58354656146/np.power(1. + n,3) - (288.82599850274386*np.power(S1,3))/n + 3456.849251298215*S3m2
    elif variation == 20:
        fit = 26928.112227857342/np.power(n,3) - (291.90305797187324*np.power(S1,3))/n + 1697.3166049143708*S3m2 - 665.9152234988633*lm11(n,S1)
    elif variation == 21:
        fit = 27506.04630707292/np.power(n,3) - (332.46674147856*np.power(S1,3))/n + 1773.3209219353353*S3m2 + 283.2262135084192*lm12(n,S1,S2)
    elif variation == 22:
        fit = 27164.01568334647/np.power(n,3) - (410.7537081255906*np.power(S1,3))/n + 1722.2439665492311*S3m2 - 119.2015925017607*lm13(n,S1,S2,S3)
    elif variation == 23:
        fit = 28283.734151920464/np.power(n,3) + (390.4853530115047*np.power(S1,2))/n - (356.68074391065085*np.power(S1,3))/n + 1869.332464998841*S3m2
    elif variation == 24:
        fit = 31017.48826899895/np.power(n,3) - (298.8574802050911*np.power(S1,3))/n + 720.3598816041747*S1m2 + 1798.7220711842397*S3m2
    else:
        fit = 216.78439081429147/(-1. + n) + 3856.767921826049/np.power(n,4) + 13855.802022791737/np.power(n,3) + 623.5284775651464/np.power(n,2) + 386.12212930499516/n + 1570.772480671127/np.power(1. + n,3) + 1370.4486038753607/np.power(1. + n,2) + 551.4507282789732/(1. + n) + 729.8759939098463/(2. + n) + (228.777848205683*np.power(S1,2))/n - (409.57994742422926*np.power(S1,3))/n + 30.014995066840612*S1m2 - 3740.159130023003*S2m2 + 3907.69267603369*S3m2 - (59.59728449377445*S3m2)/(-1. + n) + (59.59728449377445*n*S3m2)/(-1. + n) - 257.39247624856057*lm11(n,S1) + 127.55052991847907*lm12(n,S1,S2) - 48.873204785141*lm13(n,S1,S2,S3)
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
        fit = -216.50185598216353/(-1. + n) + (5.942922257778243*np.power(S1,3))/n - 11.276125560863212*S1m2 + 209.24547482131646*S2m2
    elif variation == 2:
        fit = -3424.992789569337/np.power(n,4) + (4.497110919064199*np.power(S1,3))/n + 69.36715493145202*S1m2 - 65.19066024035324*S2m2
    elif variation == 3:
        fit = -1793.0823321146702/np.power(n,3) + (10.466439610175382*np.power(S1,3))/n - 88.08845443450423*S1m2 + 127.41686995777526*S2m2
    elif variation == 4:
        fit = -894.8498968511825/np.power(n,2) + (10.215881209263914*np.power(S1,3))/n - 159.3225430245188*S1m2 + 304.02612746281477*S2m2
    elif variation == 5:
        fit = -427.5968955097106/n + (4.3409025675216*np.power(S1,3))/n + 33.07884798824976*S1m2 + 278.75039402448056*S2m2
    elif variation == 6:
        fit = -629.9033175550693/(1. + n) + (2.070872797744413*np.power(S1,3))/n + 127.47883319020481*S1m2 + 242.9519515507069*S2m2
    elif variation == 7:
        fit = -835.0523052591333/(2. + n) + (1.3569787511825873*np.power(S1,3))/n + 205.90462014157868*S1m2 + 196.22827755531864*S2m2
    elif variation == 8:
        fit = -1961.3507157568097/np.power(1. + n,2) + (6.7879600794959805*np.power(S1,3))/n - 149.94646911896686*S1m2 + 389.65463419529937*S2m2
    elif variation == 9:
        fit = -6270.091806368409/np.power(1. + n,3) + (15.260709799067673*np.power(S1,3))/n - 253.78859798513875*S1m2 + 362.43650191998154*S2m2
    elif variation == 10:
        fit = (8.45093938946135*np.power(S1,3))/n + 203.81499963952643*S1m2 + 192.5786262547217*S2m2 + 294.3121798558384*lm11(n,S1)
    elif variation == 11:
        fit = (27.975271181144024*np.power(S1,3))/n + 272.9272297977259*S1m2 + 146.31908487737235*S2m2 - 144.96082559448183*lm12(n,S1,S2)
    elif variation == 12:
        fit = (63.71051834050517*np.power(S1,3))/n + 228.08751750574626*S1m2 + 179.4863337776448*S2m2 + 55.819497666091635*lm13(n,S1,S2,S3)
    elif variation == 13:
        fit = (-254.02991937532488*np.power(S1,2))/n + (46.998805703749724*np.power(S1,3))/n + 407.67267678316904*S1m2 + 60.63003251251879*S2m2
    elif variation == 14:
        fit = -248.28454175688933/(-1. + n) + 263.2262530558056/np.power(n,3) + (5.278865453482098*np.power(S1,3))/n - (221.25799463997913*S2m2)/(-1. + n) + (221.25799463997913*n*S2m2)/(-1. + n)
    elif variation == 15:
        fit = -1916.1103405421652/np.power(n,4) - 789.943403334499/np.power(n,3) + (7.126901942835579*np.power(S1,3))/n + 19.662692663483778*S2m2
    elif variation == 16:
        fit = -489.5140579055806/np.power(n,3) - 310.8623274868738/n + (6.013182998205503*np.power(S1,3))/n + 237.43612410153995*S2m2
    elif variation == 17:
        fit = -1060.3651696443403/np.power(n,3) - 257.40088070874015/(1. + n) + (7.035712653605857*np.power(S1,3))/n + 174.62861011848457*S2m2
    elif variation == 18:
        fit = -1255.825284350709/np.power(n,3) - 250.2047609397327/(2. + n) + (7.736993248360908*np.power(S1,3))/n + 148.03467039906303*S2m2
    elif variation == 19:
        fit = -2746.3093355754777/np.power(n,3) + 3333.2662516171195/np.power(1. + n,3) + (7.917740242561949*np.power(S1,3))/n + 2.477230813827635*S2m2
    elif variation == 20:
        fit = -1251.97927524663/np.power(n,3) + (9.858216936283629*np.power(S1,3))/n + 147.0809003771629*S2m2 + 88.81534179508354*lm11(n,S1)
    elif variation == 21:
        fit = -1355.5671265199041/np.power(n,3) + (14.73862494561505*np.power(S1,3))/n + 132.0290432521235*S2m2 - 35.370693401657164*lm12(n,S1,S2)
    elif variation == 22:
        fit = -1293.5192238224065/np.power(n,3) + (25.300547889694315*np.power(S1,3))/n + 141.92372372967677*S2m2 + 15.551634890350414*lm13(n,S1,S2,S3)
    elif variation == 23:
        fit = -1474.4816162379668/np.power(n,3) - (45.13686444706024*np.power(S1,2))/n + (16.957629546978303*np.power(S1,3))/n + 115.54996689364954*S2m2
    elif variation == 24:
        fit = -1793.0823321146702/np.power(n,3) + (10.466439610175406*np.power(S1,3))/n - 88.08845443450451*S1m2 + 127.41686995777542*S2m2
    else:
        fit = -19.366099905793867/(-1. + n) - 222.54596375464592/np.power(n,4) - 626.6851209921269/np.power(n,3) - 37.28541236879927/np.power(n,2) - 30.769134291524352/n - 122.36773144797041/np.power(1. + n,3) - 81.72294648986707/np.power(1. + n,2) - 36.97100826099206/(1. + n) - 45.21904442495274/(2. + n) - (12.465282659266046*np.power(S1,2))/n + (13.604423669748037*np.power(S1,3))/n + 33.24255147579818*S1m2 + 161.28222837401603*S2m2 - (9.21908310999913*S2m2)/(-1. + n) + (9.21908310999913*n*S2m2)/(-1. + n) + 15.96364673545508*lm11(n,S1) - 7.513813291505791*lm12(n,S1,S2) + 2.973797189851752*lm13(n,S1,S2,S3)
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
