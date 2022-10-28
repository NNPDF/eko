# -*- coding: utf-8 -*-
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
    variation : str
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
        fit = 15745.62425748241/(-1. + n) + (1903.5230331588389*np.power(S1,3))/n + 48813.536840678855*S2m2 - 86016.97391082715*S3m2
    elif variation == 2:
        fit = 255436.93999852816/np.power(n,4) + (1773.6359265362535*np.power(S1,3))/n + 53974.1396409896*S2m2 - 75593.25739766566*S3m2
    elif variation == 3:
        fit = 127391.73736764734/np.power(n,3) + (1797.8383436930474*np.power(S1,3))/n + 68516.043149642*S2m2 - 95475.0396006028*S3m2
    elif variation == 4:
        fit = 62241.39486319558/np.power(n,2) + (2013.31594674906*np.power(S1,3))/n + 68428.38115369268*S2m2 - 103863.65203974594*S3m2
    elif variation == 5:
        fit = 31528.847754938073/n + (1892.3825176111927*np.power(S1,3))/n + 35364.9867245125*S2m2 - 80348.75982900734*S3m2
    elif variation == 6:
        fit = 47856.95228851117/(1. + n) + (1781.0381904795297*np.power(S1,3))/n + 19423.015991113778*S2m2 - 67746.491324651*S3m2
    elif variation == 7:
        fit = 65085.896139116005/(2. + n) + (1591.9083503164002*np.power(S1,3))/n + 6746.689691830688*S2m2 - 56679.352924251514*S3m2
    elif variation == 8:
        fit = 136799.89890214664/np.power(1. + n,2) + (2226.8625831305244*np.power(S1,3))/n + 60845.95729433034*S2m2 - 102779.69325620821*S3m2
    elif variation == 9:
        fit = 424307.3911659813/np.power(1. + n,3) + (1921.6182602873532*np.power(S1,3))/n + 80214.15184093872*S2m2 - 114459.69486613528*S3m2
    elif variation == 10:
        fit = (1046.0177002885878*np.power(S1,3))/n + 7474.183454399043*S2m2 - 56981.65868575664*S3m2 - 22923.552675601244*lm11(n,S1)
    elif variation == 11:
        fit = (-747.9260832191462*np.power(S1,3))/n - 3823.0802797942483*S2m2 - 46756.92603620328*S3m2 + 11554.25408078299*lm12(n,S1,S2)
    elif variation == 12:
        fit = (21212.77774015014*np.power(S1,2))/n - (2907.588492877621*np.power(S1,3))/n - 27813.07131726273*S2m2 - 25384.681496430592*S3m2
    elif variation == 13:
        fit = (-3375.0669771546904*np.power(S1,3))/n + 3318.042158255199*S2m2 - 53444.24165160323*S3m2 - 4382.79909331315*lm13(n,S1,S2,S3)
    elif variation == 14:
        fit = 54755.869843467335/(-1. + n) - 315616.7630493057/np.power(n,3) + (2165.3599488200225*np.power(S1,3))/n + (62584.338958215994*S3m2)/(-1. + n) - (62584.338958215994*n*S3m2)/(-1. + n)
    elif variation == 15:
        fit = 1.2035238985417755e6/np.power(n,4) - 472830.76921109675/np.power(n,3) + (1683.8055516824043*np.power(S1,3))/n - 1799.4732280361056*S3m2
    elif variation == 16:
        fit = -50400.87962059398/np.power(n,3) + 66790.93137650128/(1. + n) + (1774.3914289452387*np.power(S1,3))/n - 56776.05279709662*S3m2
    elif variation == 17:
        fit = -13914.222396866957/np.power(n,3) + 72194.83155747408/(2. + n) + (1569.4158732905269*np.power(S1,3))/n - 52441.93671808041*S3m2
    elif variation == 18:
        fit = -1.010584804100224e6/np.power(n,3) + 1.2220186257178346e6/np.power(1. + n,2) + (5630.265236435114*np.power(S1,3))/n - 160726.71663548079*S3m2
    elif variation == 19:
        fit = 873527.5448493527/np.power(n,3) - 2.485176389536297e6/np.power(1. + n,3) + (1072.857038574581*np.power(S1,3))/n + 15718.444207770523*S3m2
    elif variation == 20:
        fit = -15598.299599883458/np.power(n,3) + (953.9620938299346*np.power(S1,3))/n - 52268.39155769046*S3m2 - 25730.394390113794*lm11(n,S1)
    elif variation == 21:
        fit = 6732.578663523357/np.power(n,3) - (613.3839251706839*np.power(S1,3))/n - 49331.6497874319*S3m2 + 10943.61852384174*lm12(n,S1,S2)
    elif variation == 22:
        fit = 36781.77149498919/np.power(n,3) + (15088.019889015372*np.power(S1,2))/n - (1548.9922584166047*np.power(S1,3))/n - 45621.84592373968*S3m2
    elif variation == 23:
        fit = 142411.6912142649/np.power(n,3) + (685.2491698695229*np.power(S1,3))/n + 27834.08938919313*S1m2 - 48350.171117065765*S3m2
    elif variation == 24:
        fit = -6483.1919502412275/np.power(n,3) - (3638.325323641428*np.power(S1,3))/n - 51305.21963529444*S3m2 - 4605.847529486754*lm13(n,S1,S2,S3)
    else:
        fit = 2937.5622542062392/(-1. + n) + 60790.03493917932/np.power(n,4) - 29107.65026410144/np.power(n,3) + 2593.391452633149/np.power(n,2) + 1313.7019897890864/n - 85869.54159876314/np.power(1. + n,3) + 56617.43852583255/np.power(1. + n,2) + 4776.995152708852/(1. + n) + 5720.030320691254/(2. + n) + (1512.533234548563*np.power(S1,2))/n + (860.5068388840815*np.power(S1,3))/n + 1159.7537245497138*S1m2 + 17561.790680971935*S2m2 - 61184.72650880143*S3m2 + (2607.6807899256664*S3m2)/(-1. + n) - (2607.6807899256664*n*S3m2)/(-1. + n) - 2027.2477944047932*lm11(n,S1) + 937.4113585260304*lm12(n,S1,S2) - 374.52694261666267*lm13(n,S1,S2,S3)
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
    variation : str
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
        fit = 3785.7166189132377/(-1. + n) - (244.6533825981964*np.power(S1,3))/n - 2963.8400172742267*S2m2 + 2853.107805957077*S3m2
    elif variation == 2:
        fit = 61414.641491730144/np.power(n,4) - (275.8821081210985*np.power(S1,3))/n - 1723.0775384248207*S2m2 + 5359.279437479722*S3m2
    elif variation == 3:
        fit = 30628.764498540346/np.power(n,3) - (270.06312683778754*np.power(S1,3))/n + 1773.2287929788677*S2m2 + 579.1073675469371*S3m2
    elif variation == 4:
        fit = 14964.683461563514/np.power(n,2) - (218.25589902939427*np.power(S1,3))/n + 1752.1522405674307*S2m2 - 1437.7646853597082*S3m2
    elif variation == 5:
        fit = 7580.473213968269/n - (247.33189404761984*np.power(S1,3))/n - 6197.271657868878*S2m2 + 4215.915142027357*S3m2
    elif variation == 6:
        fit = 11506.23542429956/(1. + n) - (274.1023835976748*np.power(S1,3))/n - 10030.195812335383*S2m2 + 7245.875500079163*S3m2
    elif variation == 7:
        fit = 15648.586212999671/(2. + n) - (319.5748251680827*np.power(S1,3))/n - 13077.961868048153*S2m2 + 9906.744899797937*S3m2
    elif variation == 8:
        fit = 32890.76649300865/np.power(1. + n,2) - (166.9129330699367*np.power(S1,3))/n - 70.8880877568522*S2m2 - 1177.148738100283*S3m2
    elif variation == 9:
        fit = 102016.1230826685/np.power(1. + n,3) - (240.3027515312275*np.power(S1,3))/n + 4585.80222660763*S2m2 - 3985.3686905497766*S3m2
    elif variation == 10:
        fit = (-450.8231816718022*np.power(S1,3))/n - 12903.050719269622*S2m2 + 9834.061598925966*S3m2 - 5511.50420646659*lm11(n,S1)
    elif variation == 11:
        fit = (-882.1406501674719*np.power(S1,3))/n - 15619.249097143183*S2m2 + 12292.39158841263*S3m2 + 2777.9865045350784*lm12(n,S1,S2)
    elif variation == 12:
        fit = (5100.183003924889*np.power(S1,2))/n - (1401.3877591921307*np.power(S1,3))/n - 21387.15682013914*S2m2 + 17430.91515626977*S3m2
    elif variation == 13:
        fit = (-1513.783497714283*np.power(S1,3))/n - 13902.310762445584*S2m2 + 10684.561917950059*S3m2 - 1053.755322341623*lm13(n,S1,S2,S3)
    elif variation == 14:
        fit = 1417.10876062976/(-1. + n) + 19163.487282252743/np.power(n,3) - (260.5514874859699*np.power(S1,3))/n - (1430.3348278505848*S3m2)/(-1. + n) + (1430.3348278505848*n*S3m2)/(-1. + n)
    elif variation == 15:
        fit = 31147.788632095042/np.power(n,4) + 15094.711714218323/np.power(n,3) - (273.0143514100446*np.power(S1,3))/n + 3003.476958295951*S3m2
    elif variation == 16:
        fit = 26027.404391768934/np.power(n,3) + 1728.5820543957968/(1. + n) - (270.6699444884554*np.power(S1,3))/n + 1580.656121744779*S3m2
    elif variation == 17:
        fit = 26971.69697164435/np.power(n,3) + 1868.4376408366413/(2. + n) - (275.97481237684514*np.power(S1,3))/n + 1692.8251710670384*S3m2
    elif variation == 18:
        fit = 1177.373608113772/np.power(n,3) + 31626.441240145632/np.power(1. + n,2) - (170.87803965528423*np.power(S1,3))/n - 1109.6380297275653*S3m2
    elif variation == 19:
        fit = 49939.12505759102/np.power(n,3) - 64317.58354656146/np.power(1. + n,3) - (288.82599850274386*np.power(S1,3))/n + 3456.849251298215*S3m2
    elif variation == 20:
        fit = 26928.112227857477/np.power(n,3) - (291.90305797187193*np.power(S1,3))/n + 1697.3166049144045*S3m2 - 665.9152234988372*lm11(n,S1)
    elif variation == 21:
        fit = 27506.04630707274/np.power(n,3) - (332.46674147856623*np.power(S1,3))/n + 1773.3209219352912*S3m2 + 283.2262135084394*lm12(n,S1,S2)
    elif variation == 22:
        fit = 28283.734151920464/np.power(n,3) + (390.4853530115047*np.power(S1,2))/n - (356.68074391065085*np.power(S1,3))/n + 1869.332464998841*S3m2
    elif variation == 23:
        fit = 31017.488268999077/np.power(n,3) - (298.8574802051008*np.power(S1,3))/n + 720.3598816042988*S1m2 + 1798.7220711841446*S3m2
    elif variation == 24:
        fit = 27164.015683346614/np.power(n,3) - (410.7537081255824*np.power(S1,3))/n + 1722.2439665492548*S3m2 - 119.20159250175543*lm13(n,S1,S2,S3)
    else:
        fit = 216.78439081429156/(-1. + n) + 3856.767921826049/np.power(n,4) + 12912.581673471908/np.power(n,3) + 623.5284775651464/np.power(n,2) + 315.8530505820112/n + 1570.772480671127/np.power(1. + n,3) + 2688.2169888814283/np.power(1. + n,2) + 551.4507282789732/(1. + n) + 729.8759939098463/(2. + n) + (228.77784820568309*np.power(S1,2))/n - (405.65794826490924*np.power(S1,3))/n + 30.014995066845785*S1m2 - 3740.159130022996*S2m2 + 3803.61599177905*S3m2 - (59.59728449377437*S3m2)/(-1. + n) + (59.59728449377437*n*S3m2)/(-1. + n) - 257.39247624855943*lm11(n,S1) + 127.5505299184799*lm12(n,S1,S2) - 48.873204785140764*lm13(n,S1,S2,S3)
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
    variation : str
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
        fit = -216.50185598216368/(-1. + n) + (5.942922257778346*np.power(S1,3))/n - 11.276125560864854*S1m2 + 209.24547482131777*S2m2
    elif variation == 2:
        fit = -3424.992789569337/np.power(n,4) + (4.497110919064199*np.power(S1,3))/n + 69.36715493145202*S1m2 - 65.19066024035324*S2m2
    elif variation == 3:
        fit = -1793.0823321146702/np.power(n,3) + (10.466439610175406*np.power(S1,3))/n - 88.08845443450451*S1m2 + 127.41686995777542*S2m2
    elif variation == 4:
        fit = -894.8498968511825/np.power(n,2) + (10.215881209263914*np.power(S1,3))/n - 159.3225430245188*S1m2 + 304.02612746281477*S2m2
    elif variation == 5:
        fit = -427.5968955097113/n + (4.340902567521831*np.power(S1,3))/n + 33.07884798824544*S1m2 + 278.7503940244849*S2m2
    elif variation == 6:
        fit = -629.9033175550693/(1. + n) + (2.070872797744413*np.power(S1,3))/n + 127.47883319020481*S1m2 + 242.9519515507069*S2m2
    elif variation == 7:
        fit = -835.0523052591333/(2. + n) + (1.3569787511825924*np.power(S1,3))/n + 205.9046201415785*S1m2 + 196.22827755531884*S2m2
    elif variation == 8:
        fit = -1961.3507157568097/np.power(1. + n,2) + (6.787960079495973*np.power(S1,3))/n - 149.9464691189667*S1m2 + 389.6546341952993*S2m2
    elif variation == 9:
        fit = -6270.091806368409/np.power(1. + n,3) + (15.260709799067673*np.power(S1,3))/n - 253.78859798513875*S1m2 + 362.43650191998154*S2m2
    elif variation == 10:
        fit = (8.45093938946135*np.power(S1,3))/n + 203.81499963952643*S1m2 + 192.5786262547217*S2m2 + 294.3121798558384*lm11(n,S1)
    elif variation == 11:
        fit = (27.975271181144024*np.power(S1,3))/n + 272.9272297977259*S1m2 + 146.31908487737235*S2m2 - 144.96082559448183*lm12(n,S1,S2)
    elif variation == 12:
        fit = (-254.02991937532454*np.power(S1,2))/n + (46.9988057037496*np.power(S1,3))/n + 407.67267678317063*S1m2 + 60.63003251251638*S2m2
    elif variation == 13:
        fit = (63.71051834050536*np.power(S1,3))/n + 228.08751750574487*S1m2 + 179.4863337776457*S2m2 + 55.819497666091664*lm13(n,S1,S2,S3)
    elif variation == 14:
        fit = -248.28454175688805/(-1. + n) + 263.22625305579476/np.power(n,3) + (5.278865453482085*np.power(S1,3))/n - (221.25799463997834*S2m2)/(-1. + n) + (221.25799463997834*n*S2m2)/(-1. + n)
    elif variation == 15:
        fit = -1916.1103405422787/np.power(n,4) - 789.9434033344446/np.power(n,3) + (7.126901942835731*np.power(S1,3))/n + 19.662692663481348*S2m2
    elif variation == 16:
        fit = -1060.3651696443403/np.power(n,3) - 257.40088070874015/(1. + n) + (7.035712653605857*np.power(S1,3))/n + 174.62861011848457*S2m2
    elif variation == 17:
        fit = -1255.825284350709/np.power(n,3) - 250.2047609397327/(2. + n) + (7.736993248360908*np.power(S1,3))/n + 148.03467039906303*S2m2
    elif variation == 18:
        fit = -4346.508143070609/np.power(n,3) + 2793.0471748299365/np.power(1. + n,2) + (15.704751544035027*np.power(S1,3))/n - 246.02090466006337*S2m2
    elif variation == 19:
        fit = -2746.3093355754777/np.power(n,3) + 3333.2662516171195/np.power(1. + n,3) + (7.917740242561949*np.power(S1,3))/n + 2.477230813827635*S2m2
    elif variation == 20:
        fit = -1251.9792752466383/np.power(n,3) + (9.858216936283535*np.power(S1,3))/n + 147.08090037716144*S2m2 + 88.81534179508199*lm11(n,S1)
    elif variation == 21:
        fit = -1355.5671265199235/np.power(n,3) + (14.738624945614509*np.power(S1,3))/n + 132.02904325211964*S2m2 - 35.370693401655274*lm12(n,S1,S2)
    elif variation == 22:
        fit = -1474.4816162379668/np.power(n,3) - (45.13686444706024*np.power(S1,2))/n + (16.957629546978303*np.power(S1,3))/n + 115.54996689364954*S2m2
    elif variation == 23:
        fit = -1793.0823321146702/np.power(n,3) + (10.466439610175382*np.power(S1,3))/n - 88.08845443450423*S1m2 + 127.41686995777526*S2m2
    elif variation == 24:
        fit = -1293.5192238223985/np.power(n,3) + (25.300547889694712*np.power(S1,3))/n + 141.92372372967844*S2m2 + 15.551634890350714*lm13(n,S1,S2,S3)
    else:
        fit = -19.36609990579382/(-1. + n) - 222.54596375465067/np.power(n,4) - 787.3932078740022/np.power(n,3) - 37.28541236879927/np.power(n,2) - 17.816537312904636/n - 122.36773144797041/np.power(1. + n,3) + 34.65401912804695/np.power(1. + n,2) - 36.97100826099206/(1. + n) - 45.21904442495274/(2. + n) - (12.465282659266032*np.power(S1,2))/n + (14.008239025824277*np.power(S1,3))/n + 33.242551475797946*S1m2 + 141.13818550894914*S2m2 - (9.219083109999097*S2m2)/(-1. + n) + (9.219083109999097*n*S2m2)/(-1. + n) + 15.963646735455015*lm11(n,S1) - 7.513813291505713*lm12(n,S1,S2) + 2.9737971898517657*lm13(n,S1,S2,S3)
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
    variation : str
        |N3LO| anomalous dimension variation: "a" ,"b", "best"

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
