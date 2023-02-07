# pylint: skip-file
# fmt: off
r"""The anomalous dimension :math:`\gamma_{gg}^{(3)}`."""
import numba as nb
import numpy as np

from .....harmonics.log_functions import lm11, lm11m1


@nb.njit(cache=True)
def gamma_gg_nf3(n, sx):
    r"""Implement the part proportional to :math:`nf^3` of :math:`\gamma_{gg}^{(3)}`.

    The expression is copied exact from Eq. 3.14 of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^3}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3, S21, _, _, _, _ = sx[2]
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
def gamma_gg_nf1(n, sx, variation):
    r"""Implement the part proportional to :math:`nf^1` of :math:`\gamma_{gg}^{(3)}`.

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    common = 18143.980574437464 + 1992.766087237516/np.power(-1. + n,3) + 20005.925925925927/np.power(n,7) - 19449.679012345678/np.power(n,6) + 80274.123066115/np.power(n,5) - 11714.245609287387*S1
    if variation == 1:
        fit = 45963.268520094876/n - 51189.992910381996/(1. + n) + (5863.26605573345*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 13586.906889317255*lm11(n,S1)
    elif variation == 2:
        fit = 30156.452431280446/n - 43147.86714018369/(2. + n) + (5143.6662515144435*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 11417.515076882832*lm11(n,S1)
    elif variation == 3:
        fit = -17298.223069731634/n + 105653.13456200028/np.power(1. + n,2) + (3902.4234160578326*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 9816.003354343473*lm11(n,S1)
    elif variation == 4:
        fit = 6693.346186559492/n + 129848.6663512717/np.power(1. + n,3) + (6165.881864039302*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 16570.346446945998*lm11(n,S1)
    elif variation == 5:
        fit = -20110.063717488985/n + (36067.194950291654*S1)/np.power(n,2) + (5492.0339487451165*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 10322.724166243608*lm11(n,S1)
    elif variation == 6:
        fit = 125108.63313995973/n + (10047.104996741546*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 43219.5145607516*lm11(n,S1) + 247763.21419068478*lm11m1(n,S1)
    elif variation == 7:
        fit = 29935.996516223866/(-1. + n) - 44057.24961340775/n + (17606.5075261409*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 16238.998115450418*lm11(n,S1)
    elif variation == 8:
        fit = 73201.99904527777/np.power(n,3) + 581.902194176135/n + (13575.32523006605*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 18284.103806420848*lm11(n,S1)
    elif variation == 9:
        fit = 31440.76546384369/np.power(n,2) - 1526.9829440224726/n + (7424.37779167984*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 15158.102361356683*lm11(n,S1)
    elif variation == 10:
        fit = 15284.918092242628/(-1. + n) - 25053.069479301117/(1. + n) + (11859.20756098207*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 14941.02907939124*lm11(n,S1)
    elif variation == 11:
        fit = 12164.37706315839/(-1. + n) - 25614.89725625695/(2. + n) + (10207.893876630813*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 13376.706168215986*lm11(n,S1)
    elif variation == 12:
        fit = -19351.957542519463/(-1. + n) + 173952.01257562867/np.power(1. + n,2) - (4956.50512215596*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 5663.894309120604*lm11(n,S1)
    elif variation == 13:
        fit = 3948.1701636077623/(-1. + n) + 112723.30925835851/np.power(1. + n,3) + (7674.752192362301*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 16526.645894153065*lm11(n,S1)
    elif variation == 14:
        fit = -25139.271061083622/(-1. + n) + (66355.24598533308*S1)/np.power(n,2) - (4681.304856243236*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 5354.430747312161*lm11(n,S1)
    elif variation == 15:
        fit = 22139.52095344942/(-1. + n) + (15637.750713327814*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 23265.753915035384*lm11(n,S1) + 64526.993238533185*lm11m1(n,S1)
    elif variation == 16:
        fit = 24612.733359621925/(-1. + n) - 313416.5554676517/np.power(n,4) - (14450.489113367157*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 12394.767359948679*lm11(n,S1)
    elif variation == 17:
        fit = 390.23640352084084/(-1. + n) + 72247.76039742176/np.power(n,3) + (13627.874477268122*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 18257.444440355197*lm11(n,S1)
    elif variation == 18:
        fit = -1074.805301548193/(-1. + n) + 32569.597149364814/np.power(n,2) + (7058.8042917719695*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 15119.294483814467*lm11(n,S1)
    else:
        fit = 3494.995480370753/(-1. + n) - 17412.03085931398/np.power(n,4) + 8080.5421912610855/np.power(n,3) + 3556.1312562893613/np.power(n,2) + 6972.837951523324/n + 13476.220867201679/np.power(1. + n,3) + 15533.619285423832/np.power(1. + n,2) - 4235.725688315728/(1. + n) - 3820.1535775800357/(2. + n) + (5690.135607534708*S1)/np.power(n,2) + (6511.031727849733*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 15528.56562083664*lm11(n,S1) + 17349.455968289887*lm11m1(n,S1)
    return common + fit



@nb.njit(cache=True)
def gamma_gg_nf2(n, sx, variation):
    r"""Implement the part proportional to :math:`nf^2` of :math:`\gamma_{gg}^{(3)}`.

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^2}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    common = -423.811346198137 - 568.8888888888889/np.power(n,7) + 1725.6296296296296/np.power(n,6) - 2196.543209876543/np.power(n,5) + 440.0487580115612*S1
    if variation == 1:
        fit = -2430.638493036615/n + 2028.7214027705545/(1. + n) + (243.3886567890572*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 137.7731001470984*lm11(n,S1)
    elif variation == 2:
        fit = -1804.1952316519864/n + 1710.0022206378792/(2. + n) + (271.90726836230596*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 51.797475798299*lm11(n,S1)
    elif variation == 3:
        fit = 76.49101925214408/n - 4187.161653469534/np.power(1. + n,2) + (321.09922489832275*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 11.672372966712322*lm11(n,S1)
    elif variation == 4:
        fit = -874.3239207206218/n - 5146.059875593491/np.power(1. + n,3) + (231.39562588160055*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 256.0104223329044*lm11(n,S1)
    elif variation == 5:
        fit = 187.9276688580481/n - (1429.3866080749744*S1)/np.power(n,2) + (258.1010346977912*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 8.409586375056202*lm11(n,S1)
    elif variation == 6:
        fit = -5567.265126595396/n + (77.57805352863124*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 1312.1492229995624*lm11(n,S1) - 9819.156184057632*lm11m1(n,S1)
    elif variation == 7:
        fit = -1186.3997901318066/(-1. + n) + 1136.9836425172987/n - (222.01022206262417*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 242.8786867404777*lm11(n,S1)
    elif variation == 8:
        fit = -2901.0838592756654/np.power(n,3) - 632.1199952211956/n - (62.24958648619142*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 323.9287016030089*lm11(n,S1)
    elif variation == 9:
        fit = -1246.035605585129/np.power(n,2) - 548.5423236372875/n + (181.51990841132363*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 200.04147919616926*lm11(n,S1)
    elif variation == 10:
        fit = -808.3000072476129/(-1. + n) + 646.543541477644/(1. + n) - (73.68988644556057*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 209.38205280524372*lm11(n,S1)
    elif variation == 11:
        fit = -727.7683322791132/(-1. + n) + 661.0426079858246/(2. + n) - (31.07450159409684*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 169.0116352864771*lm11(n,S1)
    elif variation == 12:
        fit = 85.57242850810279/(-1. + n) - 4489.172488454287/np.power(1. + n,2) + (360.2725244122891*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 30.032585552420013*lm11(n,S1)
    elif variation == 13:
        fit = -515.733016177927/(-1. + n) - 2909.045841077148/np.power(1. + n,3) + (34.29818245824568*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 250.3020012247212*lm11(n,S1)
    elif variation == 14:
        fit = 234.92539226485613/(-1. + n) - (1712.427124765296*S1)/np.power(n,2) + (353.17044238259797*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 38.01889906089237*lm11(n,S1)
    elif variation == 15:
        fit = -985.1964635069794/(-1. + n) - (171.20259535184877*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 424.21788685389544*lm11(n,S1) - 1665.2454807510605*lm11m1(n,S1)
    elif variation == 16:
        fit = -1049.02255512559/(-1. + n) + 8088.328254436635/np.power(n,4) + (605.2833817638077*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 143.67078054066826*lm11(n,S1)
    elif variation == 17:
        fit = -423.91356485323354/(-1. + n) - 1864.4950036866815/np.power(n,3) - (119.33380136362118*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 294.96864866169307*lm11(n,S1)
    elif variation == 18:
        fit = -386.10529336811396/(-1. + n) - 840.522264261872/np.power(n,2) + (50.19392478804039*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 213.98254082104327*lm11(n,S1)
    else:
        fit = -320.10784455096774/(-1. + n) + 449.3515696909241/np.power(n,4) - 264.7543812756859/np.power(n,3) - 115.9198816581667/np.power(n,2) - 580.871264457534/n - 447.5058731483688/np.power(1. + n,3) - 482.01856344021223/np.power(1. + n,2) + 148.625830236011/(1. + n) + 131.72471270131686/(2. + n) - (174.545207380015*S1)/np.power(n,2) + (128.25820194833724*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 231.04446465590524*lm11(n,S1) - 638.022314711594*lm11m1(n,S1)
    return common + fit


@nb.njit(cache=True)
def gamma_gg_nf0(n, sx, variation):
    r"""Implement the part proportional to :math:`nf^0` of :math:`\gamma_{gg}^{(3)}`.

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gg}^{(3)}|_{nf^0}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    common = -68587.9129845144 - 49851.703887834694/np.power(-1. + n,4) + 213823.9810748423/np.power(-1. + n,3) - 103680./np.power(n,7) - 17280./np.power(n,6) - 627978.8224813186/np.power(n,5) + 40880.33011934297*S1
    if variation == 1:
        fit = -21320.044597892782/n - 644579.2835120695/(1. + n) + (283827.99744115997*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 119359.01638755966*lm11(n,S1)
    elif variation == 2:
        fit = -220357.89574533422/n - 543313.6381750136/(2. + n) + (274766.8685007639*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 146675.78164993884*lm11(n,S1)
    elif variation == 3:
        fit = -817902.4351105078/n + 1.3303737294124227e6/np.power(1. + n,2) + (259137.2625552471*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 166841.857477654*lm11(n,S1)
    elif variation == 4:
        fit = -515802.98866937327/n + 1.6350414517192943e6/np.power(1. + n,3) + (287638.5055074232*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 81791.84313506924*lm11(n,S1)
    elif variation == 5:
        fit = -853308.8520102579/n + (454154.5203971609*S1)/np.power(n,2) + (279153.4797877163*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 160461.27956963622*lm11(n,S1)
    elif variation == 6:
        fit = 975270.4903779445/n + (336510.47996554285*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 253771.82297170945*lm11(n,S1) + 3.1198096738019274e6*lm11m1(n,S1)
    elif variation == 7:
        fit = 376951.081423839/(-1. + n) - 1.154849427605671e6/n + (431697.72241565184*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 85964.14826670669*lm11(n,S1)
    elif variation == 8:
        fit = 921752.2686291317/np.power(n,3) - 592757.6816839123/n + (380937.47713373136*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 60212.381445797626*lm11(n,S1)
    elif variation == 9:
        fit = 395898.9818817419/np.power(n,2) - 619312.5529410484/n + (303485.3606683837*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 99574.67976149643*lm11(n,S1)
    elif variation == 10:
        fit = -7089.903435817428/(-1. + n) - 656702.8854912797/(1. + n) + (281046.782257506*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 119987.12544970344*lm11(n,S1)
    elif variation == 11:
        fit = -88886.99819047864/(-1. + n) - 671429.7804364963/(2. + n) + (237761.76832894894*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 160991.8961608483*lm11(n,S1)
    elif variation == 12:
        fit = -915008.0406742195/(-1. + n) + 4.5597122815555325e6/np.power(1. + n,2) - (159734.61779141647*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 363163.76205951767*lm11(n,S1)
    elif variation == 13:
        fit = -304254.0925573566/(-1. + n) + 2.954756602309947e6/np.power(1. + n,3) + (171361.8625293066*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 78424.18888427598*lm11(n,S1)
    elif variation == 14:
        fit = -1.066707834985954e6/(-1. + n) + (1.7393350360546543e6*S1)/np.power(n,2) - (152520.93849112795*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 371275.56704865594*lm11(n,S1)
    elif variation == 15:
        fit = 172586.18302422448/(-1. + n) + (380091.7393295146*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n + 98224.49229260809*lm11(n,S1) + 1.6914120118828905e6*lm11m1(n,S1)
    elif variation == 16:
        fit = 237415.1942920094/(-1. + n) - 8.215422725204017e6/np.power(n,4) - (408595.40856153984*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 186730.9398057005*lm11(n,S1)
    elif variation == 17:
        fit = -397516.33208317176/(-1. + n) + 1.8937924058553956e6/np.power(n,3) + (327407.91450748796*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 33055.680229707796*lm11(n,S1)
    elif variation == 18:
        fit = -435918.69694628374/(-1. + n) + 853729.6575555558/np.power(n,2) + (155216.3453516938*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 115314.34850457105*lm11(n,S1)
    else:
        fit = -134912.74667406717/(-1. + n) - 456412.3736224454/np.power(n,4) + 156419.14858247372/np.power(n,3) + 69423.8133020721/np.power(n,2) - 212241.18822144743/n + 254988.78077940227/np.power(1. + n,3) + 327227.00060933083/np.power(1. + n,2) - 72293.4538335194/(1. + n) - 67485.7454784172/(2. + n) + (121860.53091398973*S1)/np.power(n,2) + (203843.92230199964*((-1. + 2.*n - 2.*np.power(n,2))/(np.power(-1. + n,2)*np.power(n,2)) + S2))/n - 110990.45447625118*lm11(n,S1) + 267290.0936491565*lm11m1(n,S1)
    return common + fit


@nb.njit(cache=True)
def gamma_gg(n, nf, sx, variation):
    r"""Compute the |N3LO| gluon-gluon singlet anomalous dimension.

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
        |N3LO| gluon-gluon singlet anomalous dimension
        :math:`\gamma_{gg}^{(3)}(N)`

    See Also
    --------
    gamma_gg_nf0: :math:`\gamma_{gg}^{(3)}|_{nf^0}`
    gamma_gg_nf1: :math:`\gamma_{gg}^{(3)}|_{nf^1}`
    gamma_gg_nf2: :math:`\gamma_{gg}^{(3)}|_{nf^2}`
    gamma_gg_nf3: :math:`\gamma_{gg}^{(3)}|_{nf^3}`

    """
    return (
        gamma_gg_nf0(n, sx, variation)
        + nf * gamma_gg_nf1(n, sx, variation)
        + nf**2 * gamma_gg_nf2(n, sx, variation)
        + nf**3 * gamma_gg_nf3(n, sx)
    )
